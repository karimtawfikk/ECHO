// ─── Types ───────────────────────────────────────────────────────
export interface Pharaoh {
    name: string;
    dynasty: string;
    period: string;
    description: string;
    badge: string;
    icon: "crown" | "scroll" | "sparkles" | "shield" | "star";
    type?: string;
}

export interface Landmark {
    name: string;
    location: string;
    description: string;
    badge: string;
    icon: "pyramid" | "compass" | "map-pin" | "landmark" | "columns";
}

// ─── Mock Pharaohs ───────────────────────────────────────────────
export const PHARAOHS: Pharaoh[] = [
    {
        name: "Akhenaton",
        dynasty: "18th Dynasty",
        period: "New Kingdom",
        description:
            "The revolutionary pharaoh who abolished polytheism and introduced the worship of Aten, the sun disc.",
        type: "king",
        badge: "Revolutionary",
        icon: "sparkles",
    },
    {
        name: "Cleopatra VII Philopator",
        dynasty: "Ptolemaic Dynasty",
        period: "Ptolemaic Period",
        description:
            "The last active ruler of the Ptolemaic Kingdom, famed for her intelligence, political acumen, and alliances with Rome.",
        type: "queen",
        badge: "Legendary",
        icon: "crown",
    },
    {
        name: "Hatshepsut",
        dynasty: "18th Dynasty",
        period: "New Kingdom",
        description:
            "One of the most successful pharaohs, she expanded trade routes and commissioned hundreds of construction projects.",
        type: "queen",
        badge: "Royal",
        icon: "star",
    },
    {
        name: "Ramesses II",
        dynasty: "19th Dynasty",
        period: "New Kingdom",
        description:
            "Known as Ramesses the Great, he led numerous military expeditions and built monuments across Egypt.",
        type: "king",
        badge: "World-Famous",
        icon: "shield",
    },
    {
        name: "Tutankhamun",
        dynasty: "18th Dynasty",
        period: "New Kingdom",
        description:
            "The boy king whose nearly intact tomb revealed the splendours of ancient Egyptian burial traditions.",
        type: "king",
        badge: "Dynasty Icon",
        icon: "scroll",
    },
];

// ─── Mock Landmarks ──────────────────────────────────────────────
export const LANDMARKS: Landmark[] = [
    {
        name: "Pyramids of Giza",
        location: "Giza Plateau, Cairo",
        description:
            "The last surviving wonder of the ancient world, standing as an eternal testament to human ambition.",
        badge: "Ancient Wonder",
        icon: "pyramid",
    },
    {
        name: "Sphinx",
        location: "Giza Plateau, Cairo",
        description:
            "A colossal limestone statue with a lion's body and a human head, guarding the great pyramids.",
        badge: "Iconic",
        icon: "landmark",
    },
    {
        name: "Temple of Karnak",
        location: "Luxor, Upper Egypt",
        description:
            "The largest ancient religious site in the world, a vast complex of temples, pylons, and obelisks.",
        badge: "UNESCO",
        icon: "columns",
    },
    {
        name: "Temple of Luxor",
        location: "Luxor, Upper Egypt",
        description:
            "A majestic temple complex on the east bank of the Nile, dedicated to the rejuvenation of kingship.",
        badge: "Most Visited",
        icon: "compass",
    },
    {
        name: "The Great Temple of Ramesses II at Abu Simbel",
        location: "Aswan, Nubia",
        description:
            "Four colossal statues of Ramesses II carved into the cliff face, an engineering marvel relocated in 1968.",
        badge: "UNESCO",
        icon: "map-pin",
    },
];
