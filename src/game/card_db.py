"""
Card Database
Load and manage card information for Arena 1-2
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class CardType(Enum):
    """Card types"""
    TROOP = "troop"
    SPELL = "spell"
    BUILDING = "building"


class CardRole(Enum):
    """Card roles for strategic decision making"""
    TANK = "tank"
    RANGED = "ranged"
    SPLASH = "splash"
    SPELL_DAMAGE = "spell_damage"
    SPELL_TROOP = "spell_troop"
    SWARM = "swarm"
    TANK_KILLER = "tank_killer"
    SPAWNER = "spawner"
    SPLASH_AIR = "splash_air"


@dataclass
class Card:
    """Card data structure"""
    name: str
    cost: int
    type: CardType
    role: CardRole
    rarity: str
    description: str
    stats: Dict
    
    def __repr__(self) -> str:
        return f"Card({self.name}, cost={self.cost}, role={self.role.value})"


class CardDatabase:
    """Card database manager"""
    
    def __init__(self, config_path: str = "config/cards.yaml"):
        """
        Initialize card database
        
        Args:
            config_path: Path to cards.yaml
        """
        self.config_path = Path(config_path)
        self.cards: Dict[str, Card] = {}
        self.synergies: List[Dict] = []
        self.counters: List[Dict] = []
        
        self._load_database()
    
    def _load_database(self):
        """Load card database from YAML"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Card database not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Load cards
        for card_name, card_data in data['cards'].items():
            self.cards[card_name] = Card(
                name=card_name,
                cost=card_data['cost'],
                type=CardType(card_data['type']),
                role=CardRole(card_data['role']),
                rarity=card_data['rarity'],
                description=card_data['description'],
                stats=card_data['stats']
            )
        
        # Load synergies and counters
        self.synergies = data.get('synergies', [])
        self.counters = data.get('counters', [])
        
        print(f"✓ Loaded {len(self.cards)} cards from database")
    
    def get_card(self, name: str) -> Optional[Card]:
        """Get card by name"""
        return self.cards.get(name)
    
    def get_cards_by_role(self, role: CardRole) -> List[Card]:
        """Get all cards with specific role"""
        return [card for card in self.cards.values() if card.role == role]
    
    def get_cards_by_cost(self, cost: int) -> List[Card]:
        """Get all cards with specific cost"""
        return [card for card in self.cards.values() if card.cost == cost]
    
    def get_all_cards(self) -> List[Card]:
        """Get all cards"""
        return list(self.cards.values())
    
    def get_card_names(self) -> List[str]:
        """Get all card names"""
        return list(self.cards.keys())
    
    def get_synergies(self, card_name: str) -> List[Dict]:
        """
        Get synergies involving this card
        
        Returns:
            List of synergy dictionaries
        """
        return [
            syn for syn in self.synergies
            if card_name in syn['cards']
        ]
    
    def get_counters(self, card_name: str) -> List[str]:
        """
        Get cards that counter this card
        
        Returns:
            List of counter card names
        """
        for counter_info in self.counters:
            if counter_info['card'] == card_name:
                return counter_info['countered_by']
        return []
    
    def is_synergy(self, card1: str, card2: str) -> bool:
        """Check if two cards have synergy"""
        for syn in self.synergies:
            if card1 in syn['cards'] and card2 in syn['cards']:
                return True
        return False
    
    def get_synergy_bonus(self, card1: str, card2: str) -> float:
        """Get synergy bonus between two cards"""
        for syn in self.synergies:
            if card1 in syn['cards'] and card2 in syn['cards']:
                return syn.get('bonus', 0.0)
        return 0.0
    
    def get_counter_strength(self, card: str, counter: str) -> float:
        """Get counter strength"""
        for counter_info in self.counters:
            if counter_info['card'] == card and counter in counter_info['countered_by']:
                return counter_info.get('strength', 0.5)
        return 0.0
    
    def print_summary(self):
        """Print database summary"""
        print("\n=== Card Database Summary ===")
        print(f"Total cards: {len(self.cards)}")
        
        # Group by type
        by_type = {}
        for card in self.cards.values():
            card_type = card.type.value
            if card_type not in by_type:
                by_type[card_type] = []
            by_type[card_type].append(card.name)
        
        for card_type, names in by_type.items():
            print(f"\n{card_type.upper()} ({len(names)}):")
            for name in names:
                card = self.cards[name]
                print(f"  - {name:20s} | Cost: {card.cost} | Role: {card.role.value}")


if __name__ == "__main__":
    # Test card database
    db = CardDatabase()
    db.print_summary()
    
    # Test synergies
    print("\n=== Synergies ===")
    print(f"Giant + Musketeer synergy: {db.is_synergy('giant', 'musketeer')}")
    print(f"Synergy bonus: {db.get_synergy_bonus('giant', 'musketeer')}")
    
    # Test counters
    print("\n=== Counters ===")
    print(f"Skeleton Army countered by: {db.get_counters('skeleton_army')}")
