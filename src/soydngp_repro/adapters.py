"""Adapters between model variants and the corrected trainer.

The trainer only accepts plain nn.Modules with a forward() method.
Abnormal interfaces (the verbatim paper model) are wrapped here, so the
training code never sees them.
"""
from torch import nn


def unwrap_model(model):
    """Return the module that actually implements forward().

    The verbatim paper model (AlexNet, src/paper_model/soydngp_ca.py) has
    no forward() and exposes the network as .net; its .modules() override
    is also broken. Anything that must iterate modules should use the
    unwrapped network.
    """
    if hasattr(model, "net"):
        return model.net
    return model


class PaperSoyDNGP(nn.Module):
    """Thin adapter around the verbatim paper model (PAPER_MODEL_V1).

    Wraps AlexNet().net so the training system deals with a plain
    nn.Module. The verbatim source file stays untouched and remains
    auditable by diff/SHA256.
    """

    def __init__(self):
        super().__init__()
        from paper_model.soydngp_ca import AlexNet
        self.model = AlexNet().net

    def forward(self, x):
        return self.model(x)
