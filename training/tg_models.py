"""tinygrad ResNet for GPU-only training on hardware PyTorch can't reach on
this machine (an RTX 5060 Ti, visible to tinygrad's NV backend but invisible
to PyTorch on macOS -- there is no CUDA build for this platform). Vendored
rather than depended on: ``extra/`` in a tinygrad checkout is not an
importable package (``import extra.models.resnet`` raises
``ModuleNotFoundError`` unless the caller manually inserts that checkout onto
``sys.path``), and the checkout in question is a pinned local branch this
repo has no business depending on.

------------------------------------------------------------------------
Upstream attribution (required -- see below)
------------------------------------------------------------------------
Adapted from:
    https://github.com/tinygrad/tinygrad
    File:    extra/models/resnet.py
    Commit:  33cd373ad35371ccb483c9645d0c0637a04debc2
    License: MIT

    Copyright (c) 2024, the tiny corp

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.

This file (``training/tg_models.py``) is part of argus-monitor, licensed
under GPL-3.0 (see the repository's ``LICENSE`` file). The MIT license above
covers only the vendored upstream code it is adapted from; MIT -> GPL-3.0
vendoring is license-compatible as long as the original MIT notice is
preserved, which is why it's reproduced above in full rather than just
linked.

------------------------------------------------------------------------
What's changed from upstream
------------------------------------------------------------------------
* Trimmed to resnet18/34 (``BasicBlock`` only) -- this repo's classifier is a
  6-class (see ``training/build_classification_dataset.py``'s ``CLASS_NAMES``)
  small-dataset problem where a deeper ``Bottleneck`` backbone (resnet50+)
  isn't warranted; ``Bottleneck`` is still vendored (upstream needs it for
  resnet50+) but ``build_model`` only exposes the two archs this project
  actually trains.
* ``BatchNorm`` (this module's, shadowing ``tinygrad.nn.BatchNorm2d`` for
  every layer built here) pins ``weight``/``bias``/``running_mean``/
  ``running_var`` to ``dtypes.float32`` explicitly, regardless of
  ``dtypes.default_float`` -- see that class's docstring for why.
* ``build_model()`` is new: a single fail-loud entry point ("resnet18" |
  "resnet34" -> ``ResNet``) for ``training/train.py`` to call, instead of
  callers picking a constructor function directly.
* Every top-level function/method here carries this repo's standardized
  header-comment convention (see e.g. ``src/argus/detectors/classifier.py``);
  upstream's internals (``BasicBlock``/``Bottleneck``/``ResNet``) get at
  least a class + ``__call__`` docstring where a full header would be noise
  on effectively verbatim vendored code.

------------------------------------------------------------------------
The hard contract this file exists to satisfy
------------------------------------------------------------------------
``tinygrad.nn.state.get_state_dict(model)`` MUST produce exactly the same
key set (names and shapes) as
``torchvision.models.resnet{18,34}(num_classes=N).state_dict()``. This is
what lets a model trained here be loaded into a real torchvision module via
plain ``load_state_dict(strict=True)`` (no key remapping) for
``torch.onnx.export`` -- the same ONNX export path already used for the
Ultralytics-trained classifier (see ``training/export_classifier_onnx.py``).
Verified for both resnet18 (122 tensors) and resnet34 (218 tensors); see
``tests/test_tg_models.py``. Do not rename anything upstream did not rename.
"""

from __future__ import annotations

import tinygrad.nn as nn
from tinygrad import Tensor, dtypes
from tinygrad.helpers import fetch, get_child
from tinygrad.nn.state import torch_load

#: Kept as plain module-level aliases (mirroring upstream) rather than
#: importing the names directly, so nothing else in this module needs to
#: change if a caller ever wants to monkeypatch a layer implementation for
#: an experiment -- same rationale upstream documents on this comment.
Conv2d = nn.Conv2d
Linear = nn.Linear


class BatchNorm(nn.BatchNorm2d):
    """``tinygrad.nn.BatchNorm2d`` with one deliberate change from upstream:
    ``weight``, ``bias``, ``running_mean`` and ``running_var`` are always
    created as ``dtypes.float32``, even when ``dtypes.default_float`` is
    ``dtypes.float16`` (the case we train under on the RTX 5060 Ti, for
    speed). BatchNorm's running-variance accumulation (an exponential moving
    average across many steps) and its ``1/sqrt(var + eps)`` division are
    both numerically fragile in half precision -- small variances and a
    small ``eps`` can underflow or lose precision badly in fp16, silently
    corrupting normalization for the rest of training. tinygrad's own
    ``examples/hlb_cifar10.py`` hits the identical problem for its fp16 run
    and solves it the same way (see its ``UnsyncedBatchNorm``): pin the
    normalization's parameters and internal math to float32, and only cast
    back down to the activation's own dtype (whatever ``x.dtype`` is -- fp16
    in our case) on the way out, so the rest of the network still gets the
    fp16 speedup everywhere except inside this one op.

    This does NOT affect ``get_state_dict`` key parity: parity is about
    tensor *names and shapes* matching torchvision's state dict, not dtype
    (torchvision's own ``BatchNorm2d`` buffers are fp32 regardless of the
    model's overall training precision, so this is actually the *more*
    faithful match, not a deviation from the export target).
    """

    # None __init__(int sz, float eps, bool affine, bool track_running_stats, float momentum)
    # Inputs: int sz - number of channels (features) this BatchNorm normalizes
    #         float eps - added to variance before the rsqrt division, default 1e-5
    #         bool affine - whether to learn a per-channel scale/shift, default True
    #         bool track_running_stats - whether to maintain running_mean/running_var buffers
    #                 for eval-mode use, default True
    #         float momentum - running-stats exponential-average momentum, default 0.1
    # Outputs: None
    # Description: Builds this layer's affine parameters and running-stat buffers exactly as
    #              ``tinygrad.nn.BatchNorm`` does, except every one of them is forced to
    #              ``dtypes.float32`` (see class docstring) instead of inheriting whatever
    #              ``dtypes.default_float`` happens to be at construction time.
    # Side Effects: Mutates the new instance's state (weight, bias, num_batches_tracked,
    #               running_mean, running_var).
    def __init__(
        self,
        sz: int,
        eps: float = 1e-5,
        affine: bool = True,
        track_running_stats: bool = True,
        momentum: float = 0.1,
    ) -> None:
        self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum
        self.weight: Tensor | None = Tensor.ones(sz, dtype=dtypes.float32) if affine else None
        self.bias: Tensor | None = Tensor.zeros(sz, dtype=dtypes.float32) if affine else None
        self.num_batches_tracked = Tensor.zeros(dtype="long").is_param_(False)
        if track_running_stats:
            self.running_mean = Tensor.zeros(sz, dtype=dtypes.float32).is_param_(False)
            self.running_var = Tensor.ones(sz, dtype=dtypes.float32).is_param_(False)

    # Tensor __call__(Tensor x)
    # Inputs: Tensor x - activations to normalize, any float dtype (e.g. float16 under fp16
    #                 training)
    # Outputs: Tensor - normalized activations, cast back to x's original dtype
    # Description: Runs the inherited ``tinygrad.nn.BatchNorm.__call__`` (mean/var stats,
    #              running-stat updates, the affine transform) with x upcast to float32 first,
    #              so the whole computation -- not just the stored parameters -- happens in full
    #              precision, then casts the result back down to x's original dtype. Without this
    #              cast, ``Tensor.batchnorm``'s elementwise ops between a float16 ``x`` and this
    #              class's float32 weight/bias/mean/invstd would silently upcast the *output* to
    #              float32 anyway (tinygrad promotes mixed-dtype binary ops), quietly widening
    #              every downstream activation for the rest of the network and defeating the
    #              point of training in fp16. Casting back here keeps that promotion strictly
    #              local to this op.
    # Side Effects: Inherited from ``tinygrad.nn.BatchNorm.__call__``: in training mode, updates
    #               ``running_mean``/``running_var``/``num_batches_tracked`` in place.
    def __call__(self, x: Tensor) -> Tensor:
        out_dtype = x.dtype
        return super().__call__(x.cast(dtypes.float32)).cast(out_dtype)


class BasicBlock:
    """Vendored verbatim from upstream (see module docstring). The two-conv
    residual block used by resnet18/34: ``conv1 -> bn1 -> relu -> conv2 ->
    bn2``, added to a (possibly downsampled) identity shortcut, then
    ``relu``'d again. ``downsample`` is a ``[Conv2d, BatchNorm]`` pair (empty
    list = identity shortcut) inserted whenever this block changes spatial
    stride or channel count, so the shortcut's shape matches the main path's.
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, groups: int = 1, base_width: int = 64) -> None:
        assert groups == 1 and base_width == 64, "BasicBlock only supports groups=1 and base_width=64"
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.downsample = []
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = [
                Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                BatchNorm(self.expansion * planes),
            ]

    # Tensor __call__(Tensor x)
    # Inputs: Tensor x - block input activations
    # Outputs: Tensor - block output activations, same spatial/channel shape contract as
    #          upstream (stride/expansion determine how they differ from x's)
    # Description: conv1->bn1->relu->conv2->bn2, added to x passed through the (possibly empty,
    #              i.e. identity) downsample path, then relu'd.
    # Side Effects: None beyond whatever BatchNorm.__call__ does (running-stat updates in
    #               training mode).
    def __call__(self, x: Tensor) -> Tensor:
        out = self.bn1(self.conv1(x)).relu()
        out = self.bn2(self.conv2(out))
        out = out + x.sequential(self.downsample)
        out = out.relu()
        return out


class Bottleneck:
    """Vendored verbatim from upstream (see module docstring). The three-conv
    residual block used by resnet50/101/152 -- not exposed via
    ``build_model`` in this project (see module docstring's "what's
    changed"), but kept because ``ResNet.__init__`` still needs it available
    for those depths. NOTE (upstream's): ``stride_in_1x1=False`` by default,
    the "v1.5" variant.
    """

    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        stride_in_1x1: bool = False,
        groups: int = 1,
        base_width: int = 64,
    ) -> None:
        width = int(planes * (base_width / 64.0)) * groups
        # NOTE (upstream's): the original implementation places stride at the first
        # convolution (self.conv1); control with stride_in_1x1.
        self.conv1 = Conv2d(in_planes, width, kernel_size=1, stride=stride if stride_in_1x1 else 1, bias=False)
        self.bn1 = BatchNorm(width)
        self.conv2 = Conv2d(
            width, width, kernel_size=3, padding=1, stride=1 if stride_in_1x1 else stride, groups=groups, bias=False
        )
        self.bn2 = BatchNorm(width)
        self.conv3 = Conv2d(width, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(self.expansion * planes)
        self.downsample = []
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = [
                Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                BatchNorm(self.expansion * planes),
            ]

    # Tensor __call__(Tensor x)
    # Inputs: Tensor x - block input activations
    # Outputs: Tensor - block output activations
    # Description: conv1->bn1->relu->conv2->bn2->relu->conv3->bn3, added to x passed through the
    #              (possibly empty, i.e. identity) downsample path, then relu'd.
    # Side Effects: None beyond whatever BatchNorm.__call__ does (running-stat updates in
    #               training mode).
    def __call__(self, x: Tensor) -> Tensor:
        out = self.bn1(self.conv1(x)).relu()
        out = self.bn2(self.conv2(out)).relu()
        out = self.bn3(self.conv3(out))
        out = out + x.sequential(self.downsample)
        out = out.relu()
        return out


class ResNet:
    """Vendored verbatim from upstream (see module docstring), aside from
    this module's ``BatchNorm`` shadowing ``tinygrad.nn.BatchNorm2d``
    everywhere ``BasicBlock``/``Bottleneck``/this class construct one.
    Standard torchvision-shape ResNet: a 7x7 stride-2 stem, four
    ``_make_layer`` stages, global-average-pool, and (when ``num_classes``
    is given) a final ``Linear`` head. With ``num_classes=None`` this is a
    feature extractor (``forward`` returns the four stage outputs instead of
    logits) -- unused by ``build_model`` but kept for parity with upstream.
    """

    def __init__(
        self,
        num: int,
        num_classes: int | None = None,
        groups: int = 1,
        width_per_group: int = 64,
        stride_in_1x1: bool = False,
    ) -> None:
        self.num = num
        self.block = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck}[num]

        self.num_blocks = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[num]

        self.in_planes = 64

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, bias=False, padding=3)
        self.bn1 = BatchNorm(64)
        self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1, stride_in_1x1=stride_in_1x1)
        self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2, stride_in_1x1=stride_in_1x1)
        self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2, stride_in_1x1=stride_in_1x1)
        self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2, stride_in_1x1=stride_in_1x1)
        self.fc = Linear(512 * self.block.expansion, num_classes) if num_classes is not None else None

    # list _make_layer(type block, int planes, int num_blocks, int stride, bool stride_in_1x1)
    # Inputs: type block - BasicBlock or Bottleneck
    #         int planes - base channel width for this stage
    #         int num_blocks - how many residual blocks this stage contains
    #         int stride - stride of this stage's first block (later blocks are stride 1)
    #         bool stride_in_1x1 - Bottleneck-only stride placement flag, ignored by BasicBlock
    # Outputs: list - the stage's block instances, in forward order
    # Description: Builds one ResNet stage as a list of block instances, threading
    #              self.in_planes through so each subsequent block's input-channel count is
    #              correct.
    # Side Effects: Mutates self.in_planes.
    def _make_layer(self, block, planes: int, num_blocks: int, stride: int, stride_in_1x1: bool) -> list:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            if block == Bottleneck:
                layers.append(block(self.in_planes, planes, stride, stride_in_1x1, self.groups, self.base_width))
            else:
                layers.append(block(self.in_planes, planes, stride, self.groups, self.base_width))
            self.in_planes = planes * block.expansion
        return layers

    # Tensor | list[Tensor] forward(Tensor x)
    # Inputs: Tensor x - input image batch, shape (N, 3, H, W)
    # Outputs: Tensor | list[Tensor] - class logits of shape (N, num_classes) when this ResNet
    #          was built with a num_classes (the only mode build_model uses); the four per-stage
    #          feature maps instead when it was built as a feature extractor (num_classes=None)
    # Description: Stem (conv1/bn1/relu/maxpool) followed by the four residual stages, then
    #              either global-average-pool + fc (classifier mode) or the raw per-stage
    #              features (feature-extractor mode, num_classes=None).
    # Side Effects: None beyond whatever BatchNorm.__call__ does (running-stat updates in
    #               training mode).
    def forward(self, x: Tensor) -> Tensor:
        is_feature_only = self.fc is None
        if is_feature_only:
            features = []
        out = self.bn1(self.conv1(x)).relu()
        out = out.pad([1, 1, 1, 1]).max_pool2d((3, 3), 2)
        out = out.sequential(self.layer1)
        if is_feature_only:
            features.append(out)
        out = out.sequential(self.layer2)
        if is_feature_only:
            features.append(out)
        out = out.sequential(self.layer3)
        if is_feature_only:
            features.append(out)
        out = out.sequential(self.layer4)
        if is_feature_only:
            features.append(out)
        if not is_feature_only:
            out = out.mean([2, 3])
            out = self.fc(out.cast(dtypes.float32))
            return out
        return features

    # Tensor __call__(Tensor x)
    # Inputs: Tensor x - input image batch, shape (N, 3, H, W)
    # Outputs: Tensor - see forward()
    # Description: Thin __call__ -> forward() forwarding, matching upstream and tinygrad's usual
    #              module-call convention.
    # Side Effects: Same as forward().
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    # None load_from_pretrained()
    # Inputs: None (operates on self: self.num, self.groups, self.base_width, and every
    #         parameter tensor reachable from self)
    # Outputs: None
    # Description: Downloads (via tinygrad.helpers.fetch, which caches locally) the matching
    #              torchvision ImageNet checkpoint for this ResNet's (depth, groups, base_width),
    #              loads it with tinygrad.nn.state.torch_load, and assigns each tensor onto the
    #              correspondingly-named parameter found via get_child. A checkpoint's ``fc.*``
    #              entries are skipped if this ResNet has no fc (num_classes=None) or if the
    #              checkpoint's fc shape doesn't match this model's num_classes (transfer
    #              learning onto a different class count) -- both are expected, not errors.
    #              Requires network access; NOT exercised by tests/test_tg_models.py, which must
    #              run offline.
    # Side Effects: Makes an HTTP request (via fetch) to download.pytorch.org unless the
    #               checkpoint is already cached; reads the cached/downloaded file from disk;
    #               mutates every matched parameter tensor in place via .assign(). Raises
    #               AssertionError if a non-BN, non-downsample parameter's shape doesn't match
    #               the checkpoint's.
    def load_from_pretrained(self) -> None:
        model_urls = {
            (18, 1, 64): "https://download.pytorch.org/models/resnet18-5c106cde.pth",
            (34, 1, 64): "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
            (50, 1, 64): "https://download.pytorch.org/models/resnet50-19c8e357.pth",
            (50, 32, 4): "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
            (101, 1, 64): "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
            (152, 1, 64): "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
        }

        self.url = model_urls[(self.num, self.groups, self.base_width)]
        for k, dat in torch_load(fetch(self.url)).items():
            try:
                obj: Tensor = get_child(self, k)
            except AttributeError as e:
                if "fc." in k and self.fc is None:
                    continue
                raise e

            if "fc." in k and obj.shape != dat.shape:
                print("skipping fully connected layer")
                continue  # Skip FC if transfer learning

            if "bn" not in k and "downsample" not in k:
                assert obj.shape == dat.shape, (k, obj.shape, dat.shape)
            obj.assign(dat.to(obj.device).cast(obj.dtype).reshape(obj.shape))


# ResNet ResNet18(int num_classes)
# Inputs: int num_classes - number of output classes, default 1000 (ImageNet)
# Outputs: ResNet - an untrained (random-init) resnet18
# Description: Constructs a depth-18 ResNet (BasicBlock, [2,2,2,2] blocks per stage). Vendored
#              from upstream's module-level lambda as a real ``def`` so it carries this repo's
#              header-comment convention.
# Side Effects: None (pure construction; no I/O, no network).
def ResNet18(num_classes: int = 1000) -> ResNet:
    return ResNet(18, num_classes=num_classes)


# ResNet ResNet34(int num_classes)
# Inputs: int num_classes - number of output classes, default 1000 (ImageNet)
# Outputs: ResNet - an untrained (random-init) resnet34
# Description: Constructs a depth-34 ResNet (BasicBlock, [3,4,6,3] blocks per stage). Vendored
#              from upstream's module-level lambda as a real ``def`` so it carries this repo's
#              header-comment convention.
# Side Effects: None (pure construction; no I/O, no network).
def ResNet34(num_classes: int = 1000) -> ResNet:
    return ResNet(34, num_classes=num_classes)


#: arch name -> zero-arg-except-num_classes constructor. The only two depths this project
#: trains (see module docstring's "what's changed") -- resnet50+ (Bottleneck) is deliberately
#: not exposed here even though the class is vendored and available.
_ARCH_BUILDERS = {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
}


# ResNet build_model(str arch, int num_classes, bool pretrained)
# Inputs: str arch - architecture name, must be one of _ARCH_BUILDERS' keys ("resnet18",
#                 "resnet34")
#         int num_classes - number of output classes for the final fc layer
#         bool pretrained - whether to load torchvision ImageNet weights via
#                 ResNet.load_from_pretrained after construction, default True
# Outputs: ResNet - a constructed (and optionally pretrained-weight-loaded) ResNet matching arch
# Description: Single fail-loud entry point training/train.py uses to build a tinygrad ResNet
#              backbone by name, instead of callers reaching for ResNet18/ResNet34 directly. An
#              unrecognized arch raises ValueError naming both the bad value and the valid
#              options -- never silently falls back to a default architecture.
# Side Effects: When pretrained=True, has all the side effects of ResNet.load_from_pretrained
#               (network access to download.pytorch.org, disk cache read, in-place tensor
#               assignment). None otherwise.
def build_model(arch: str, num_classes: int, pretrained: bool = True) -> ResNet:
    try:
        builder = _ARCH_BUILDERS[arch]
    except KeyError:
        valid = sorted(_ARCH_BUILDERS.keys())
        raise ValueError(f"unknown arch {arch!r}; valid options are {valid}") from None

    model = builder(num_classes=num_classes)
    if pretrained:
        model.load_from_pretrained()
    return model
