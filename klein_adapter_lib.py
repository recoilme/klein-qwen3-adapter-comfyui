#!/usr/bin/env python3
"""Схема адаптера и его загрузка — в одном месте.

Раньше класс адаптера был скопирован в train_adapter.py и test_klein_adapter.py
с комментарием «схемы обязаны совпадать»: любая правка в одном файле ломала
второй молча, потому что веса грузятся по именам и совпадают по shape.
Теперь обучение, A/B-тест, экспорт в safetensors и пример с klein ходят сюда.

Схема: MLP поточечно (каждый токен независимо)
    in -> [norm] -> hidden -> GELU(tanh) -> ... -> out
norm="ln-per-layer" — каждый срез слоя ТЕ нормируется отдельно, затем склейка
(нормы скрытых состояний растут с глубиной в 40 раз: 13 у слоя 2, 557 у слоя 27,
поэтому без пер-слойной нормировки ранние слои не влияют на выход).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PerLayerNorm(nn.Module):
    """Нормировка каждого среза слоя отдельно + общая нормировка склейки.

    window=0 — поточечный режим (как было). window=k>0 добавляет в склейку соседей
    x_{i-k}..x_{i+k}: понятие у ТЕ бывает разорвано на куски («Ġbulld»+«og»), и по одному
    куску его не собрать. Соседние блоки в первом линейном слое инициализируются нулями
    (см. transfer_pointwise_to_window), поэтому на старте функция та же самая.
    """

    def __init__(self, in_dim, n_slices, window=0):
        super().__init__()
        assert in_dim % n_slices == 0, f"{in_dim} не делится на {n_slices}"
        self.n, self.d, self.window = n_slices, in_dim // n_slices, window
        self.per = nn.LayerNorm(self.d, elementwise_affine=False)
        self.all = nn.LayerNorm(in_dim)

    def forward(self, x):
        b, l, _ = x.shape
        y = self.all(self.per(x.view(b, l, self.n, self.d)).reshape(b, l, -1))
        if self.window:                     # [x_{i-k} | ... | x_i | ... | x_{i+k}], края — нули
            yp = F.pad(y, (0, 0, self.window, self.window))
            y = torch.cat([yp[:, i:i + l] for i in range(2 * self.window + 1)], dim=-1)
        return y


def transfer_pointwise_to_window(sd, in_dim, hidden, window):
    """Веса поточечной MLP -> в оконную: центр на месте, блоки соседей — нули.

    На старте новая функция побитово равна старой (нулевые блоки ничего не добавляют),
    дальше дообучение само решает, сколько брать от соседей.
    """
    w = sd["1.weight"]
    assert w.shape[1] == in_dim, f"ожидал {in_dim} на входе, получил {w.shape[1]}"
    new = torch.zeros(hidden, in_dim * (2 * window + 1), dtype=w.dtype, device=w.device)
    new[:, window * in_dim:(window + 1) * in_dim] = w
    out = dict(sd)
    out["1.weight"] = new
    return out


def build_adapter(in_dim, out_dim, hidden=4096, layers=2, norm="none", n_slices=1, window=0,
                  attention=0, attn_dim=1024, attn_heads=8, max_len=256):
    """MLP-проектор: `layers` линейных слоёв, `layers-1` GELU(tanh) между ними.

    window>0 — вариант с соседними токенами (см. PerLayerNorm): вход первого слоя шире
    в (2*window+1) раз.
    attention>0 — добавить residual-ветку self-attention на `attention` блоков
    (выход ветки инициализирован нулями, поэтому на старте функция та же, что у чистого MLP).
    """
    if layers < 2:
        raise SystemExit("нужно хотя бы 2 линейных слоя")
    mods = []
    if norm == "ln":
        mods.append(nn.LayerNorm(in_dim))
    elif norm == "ln-per-layer":
        mods.append(PerLayerNorm(in_dim, n_slices, window=window))
    elif norm == "rms":
        mods.append(nn.RMSNorm(in_dim))
    elif norm != "none":
        raise SystemExit(f"неизвестный --norm: {norm}")
    mods += [nn.Linear(in_dim * (2 * window + 1), hidden), nn.GELU(approximate="tanh")]
    for _ in range(layers - 2):
        mods += [nn.Linear(hidden, hidden), nn.GELU(approximate="tanh")]
    mods.append(nn.Linear(hidden, out_dim))
    if attention:
        return AttnAdapter(mods, AttnMixer(in_dim, out_dim, attn_dim, attn_heads,
                                           attention, max_len))
    return nn.Sequential(*mods)


class AttnBlock(nn.Module):
    """Pre-norm self-attention + FFN. Без причинности: текст не автогенеративный,
    а DiT всё равно видит всю последовательность целиком."""

    def __init__(self, d_model, n_heads, ffn_mult=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.n2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * ffn_mult),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(d_model * ffn_mult, d_model))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ffn(self.n2(x))


class AttnMixer(nn.Module):
    """Ветка внимания: своя нормировка -> проекция в d_model -> N блоков -> проекция в out_dim.

    Последняя проекция инициализирована нулями: на старте ветка не добавляет ничего (см. AttnAdapter).
    """

    def __init__(self, in_dim, out_dim, d_model=1024, n_heads=8, blocks=2, max_len=256):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.inp = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = nn.ModuleList([AttnBlock(d_model, n_heads) for _ in range(blocks)])
        self.out = nn.Linear(d_model, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = self.inp(self.norm(x))
        h = h + self.pos[:, :h.shape[1]]
        for b in self.blocks:
            h = b(h)
        return self.out(h)


class AttnAdapter(nn.Module):
    """Поточечная MLP (self.mlp) + residual-ветка внимания (self.attn).

    Не подкласс nn.Sequential: Sequential обходит все свои подмодули, включая ветку, и она
    получила бы на вход 7680 вместо 6144. Ключи MLP получают префикс "mlp.", ветка — "attn.";
    старый чекпоинт грузится с переносом префикса (см. train_adapter.load_into).
    """

    def __init__(self, mods, attn):
        super().__init__()
        self.mlp = nn.Sequential(*mods)
        self.attn = attn

    def forward(self, x):
        return self.mlp(x) + self.attn(x)


def infer_arch(state_dict):
    """Размеры и тип нормировки прямо из весов — чтобы примеру не нужны были флаги.

    Нормировка опознаётся по ключам: `0.per.weight` — ln-per-layer, `0.weight`+`0.bias`
    (1-D) — LayerNorm, `0.weight` (1-D) без bias — RMSNorm, `0.weight` (2-D) — нормировки нет.
    """
    mlp = {k[len("mlp."):]: v for k, v in state_dict.items() if k.startswith("mlp.")}
    if not mlp:                                   # старый поточечный чекпоинт: без префикса
        mlp = {k: v for k, v in state_dict.items() if not k.startswith("attn.")}
    linears = sorted((int(k.split(".")[0]), k) for k, v in mlp.items()
                     if k.endswith(".weight") and v.dim() == 2 and k.split(".")[0].isdigit())
    if not linears:
        raise SystemExit("в весах нет линейных слоёв")
    first, last = mlp[linears[0][1]], mlp[linears[-1][1]]
    if "0.per.weight" in mlp or "0.all.weight" in mlp:
        norm = "ln-per-layer"
        in_dim = mlp["0.all.weight" if "0.all.weight" in mlp else "0.per.weight"].shape[0]
    elif any(k.startswith("0.") and v.dim() == 1 for k, v in mlp.items()):
        norm = "ln" if "0.bias" in mlp else "rms"
        in_dim = first.shape[1]
    else:
        norm = "none"
        in_dim = first.shape[1]
    attention = len({k.split(".")[2] for k in state_dict if k.startswith("attn.blocks.")})
    arch = {"in_dim": in_dim, "out_dim": last.shape[0], "hidden": first.shape[0],
            "proj_layers": len(linears), "norm": norm, "attention": attention}
    if attention:
        arch["attn_dim"] = state_dict["attn.out.weight"].shape[1]
    return arch


def load_adapter(path, device="cuda", dtype=None, n_slices=None):
    """Веса -> (adapter в eval, meta). Понимает .pt (голые веса или полный чекпоинт) и .safetensors.

    n_slices (число слоёв ТЕ в склейке) нельзя вытащить из весов: у PerLayerNorm
    нормировка без обучаемых параметров. Поэтому он берётся из meta, а если её нет —
    из аргумента; при norm="ln-per-layer" и отсутствии обоих это ошибка, а не догадка.
    """
    meta = {}
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        from safetensors import safe_open
        sd = load_file(path)
        with safe_open(path, framework="pt") as f:
            meta = dict(f.metadata() or {})
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:      # полный чекпоинт обучения
            meta = {k: v for k, v in sd.items() if k != "state_dict"}
            sd = sd["state_dict"]

    arch = infer_arch(sd)
    if meta.get("student_layers"):
        layers = [int(x) for x in str(meta["student_layers"]).split(",")]
        n_slices = n_slices or len(layers)
    if n_slices is None:
        if arch["norm"] == "ln-per-layer":
            raise SystemExit("нужен n_slices: в весах его нет, а norm=ln-per-layer без него "
                             "загрузится с неверной формой и будет считать мусор")
        n_slices = 1
    adapter = build_adapter(arch["in_dim"], arch["out_dim"], arch["hidden"],
                            arch["proj_layers"], arch["norm"], n_slices=n_slices,
                            attention=arch["attention"],
                            attn_dim=arch.get("attn_dim", 1024),
                            attn_heads=int(meta.get("attn_heads", 8)))
    adapter.load_state_dict(sd)
    if dtype is not None:
        adapter = adapter.to(dtype)
    adapter = adapter.to(device).eval()
    meta.update(arch | {"n_slices": n_slices})
    return adapter, meta


def save_safetensors(adapter, path, meta=None):
    """Сохранить адаптер в safetensors вместе с метаданными (строки, как требует формат)."""
    from safetensors.torch import save_file
    md = {k: ("".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v))
          for k, v in (meta or {}).items()}
    save_file({k: v.contiguous() for k, v in adapter.state_dict().items()}, path, metadata=md)
