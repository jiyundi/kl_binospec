"""
doublet_utils.py  —  Doublet-line constants and parameter-expansion helpers
===========================================================================
Centralises everything that knows about doublet emission-line pairs
(currently O2 = [O II] λλ3726,3729) so no other module needs to
hard-code the primary ↔ secondary parameter mapping.

Public API
----------
DOUBLET_LINES       : list[str]
DOUBLET_PAR_MAP     : dict[str, str]
DoubletExpander     : stateless helper class
deduplicate_ordered : preserve-order dedup utility
"""

# ── Doublet configuration ─────────────────────────────────────────────────────

DOUBLET_LINES: list[str] = ['O2']

# primary parameter name  →  doublet-component counterpart
DOUBLET_PAR_MAP: dict[str, str] = {
    'v_0':    'v_0_2',
    'dx_vel': 'dx_vel_2',
    'dy_vel': 'dy_vel_2',
    'I01':    'I02',
    'f1_1':   'f2_1',
    'f1_2':   'f2_2',
}

# Ordered lists retained for any index-based legacy look-ups
DOUBLET_PARS_TWIN: list[str] = list(DOUBLET_PAR_MAP.keys())
DOUBLET_PARS:      list[str] = list(DOUBLET_PAR_MAP.values())


# ── Helper class ──────────────────────────────────────────────────────────────

class DoubletExpander:
    """
    Stateless helper for resolving doublet partner parameter names and labels.

    All methods are static; the class is a namespace rather than a factory.
    """

    @staticmethod
    def partner_name_with_spec(pname_w_spec: str) -> str | None:
        """
        Return the doublet-partner parameter name (preserving any ``_specN``
        suffix), or ``None`` if *pname_w_spec* is not a primary doublet param.

        Examples
        --------
        >>> DoubletExpander.partner_name_with_spec('I01_spec2')
        'I02_spec2'
        >>> DoubletExpander.partner_name_with_spec('v_0')
        'v_0_2'
        >>> DoubletExpander.partner_name_with_spec('bkg_level')
        None
        """
        if '_spec' in pname_w_spec:
            base, _, spec_suffix = pname_w_spec.partition('_spec')
            partner = DOUBLET_PAR_MAP.get(base)
            return f'{partner}_spec{spec_suffix}' if partner else None
        return DOUBLET_PAR_MAP.get(pname_w_spec)

    @staticmethod
    def partner_latex(latex_name: str) -> str:
        """
        Append a ``_{(2)}`` subscript to a LaTeX parameter-name string.

        Example
        -------
        >>> DoubletExpander.partner_latex(r'$v_0$')
        '$v_0\\,_{(2)}$'
        """
        return latex_name[:-1] + r'\,_{(2)}$'


# ── Utility ───────────────────────────────────────────────────────────────────

def deduplicate_ordered(seq) -> list:
    """Return a list with duplicates removed, preserving first-occurrence order."""
    seen: set = set()
    return [x for x in seq if not (x in seen or seen.add(x))]
