"""Card representation for FreeCell."""

SUIT_SYMBOLS = ['♣', '♦', '♥', '♠']
SUIT_NAMES   = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
SUIT_COLORS  = ['black', 'red', 'red', 'black']
RANK_SYMBOLS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']


class Card:
    """An immutable playing card with rank (1-13) and suit (0-3)."""

    __slots__ = ('rank', 'suit')

    def __init__(self, rank: int, suit: int):
        self.rank = rank   # 1 = Ace … 13 = King
        self.suit = suit   # 0 = Clubs, 1 = Diamonds, 2 = Hearts, 3 = Spades

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def is_red(self) -> bool:
        return self.suit in (1, 2)

    @property
    def color(self) -> str:
        return SUIT_COLORS[self.suit]

    @property
    def rank_str(self) -> str:
        return RANK_SYMBOLS[self.rank - 1]

    @property
    def suit_symbol(self) -> str:
        return SUIT_SYMBOLS[self.suit]

    # ── Game logic ───────────────────────────────────────────────────────────

    def can_stack_on(self, other: 'Card') -> bool:
        """True if this card can be placed on *other* in a cascade
        (descending rank, alternating colour)."""
        return (self.rank == other.rank - 1) and (self.is_red != other.is_red)

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_int(self) -> int:
        """Compact integer id: rank * 4 + suit (0‥51)."""
        return (self.rank - 1) * 4 + self.suit

    @staticmethod
    def from_int(n: int) -> 'Card':
        return Card(n // 4 + 1, n % 4)

    # ── Dunder methods ───────────────────────────────────────────────────────

    def __str__(self)  -> str: return f"{self.rank_str}{self.suit_symbol}"
    def __repr__(self) -> str: return str(self)

    def __eq__(self, other) -> bool:
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return (self.rank - 1) * 4 + self.suit

    def __lt__(self, other: 'Card') -> bool:
        return (self.suit, self.rank) < (other.suit, other.rank)
