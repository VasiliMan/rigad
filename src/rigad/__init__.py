"""RIGAD — Researcher Interest Group Allocation and Discovery.

Point it at a folder of draft papers and it suggests where to submit them, who
to talk to across the EUTOPIA alliance, and — given enough drafts — how to
group them.

    from rigad import analyse

    result = analyse("~/my-drafts")
    result.show()
"""

from .analyse import Analysis, DraftResult, analyse

__all__ = ["analyse", "Analysis", "DraftResult"]
__version__ = "0.2.0"
