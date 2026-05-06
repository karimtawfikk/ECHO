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
            "Akhenaten was a pharaoh of the Eighteenth Dynasty who reigned from the first year of his rule until his death in his seventeenth regnal year, around 1334 or 1335 BCE.",
        type: "king",
        badge: "Revolutionary",
        icon: "sparkles",
    },
    {
        name: "Cleopatra VII Philopator",
        dynasty: "Ptolemaic Dynasty",
        period: "Ptolemaic Period",
        description:
            "Cleopatra VII, the last active ruler of Ptolemaic Egypt (69–30 BCE), wielded political genius to maintain sovereignty against Rome through alliances with Julius Caesar and Mark Antony. Her death marked the end of ancient Egypt’s independence and its annexation as a Roman province.",
        type: "queen",
        badge: "Legendary",
        icon: "crown",
    },
    {
        name: "Hatshepsut",
        dynasty: "18th Dynasty",
        period: "New Kingdom",
        description:
            "Hatshepsut reigned as pharaoh (c. 1479–1458 BCE) in the 18th Dynasty, one of ancient Egypt’s few female rulers, and commissioned massive building projects including her mortuary temple at Deir el-Bahari. She emphasized divine legitimacy through propaganda and trade expeditions, notably to Punt, enriching Egypt economically and culturally.",
        type: "queen",
        badge: "Royal",
        icon: "star",
    },
    {
        name: "Ramesses II",
        dynasty: "19th Dynasty",
        period: "New Kingdom",
        description:
            "Ramesses II was the third pharaoh of the nineteenth Dynasty and reigned from 1279 BCE until his death in 1213 BCE, marking one of the longest and most influential reigns in Egyptian history.",
        type: "king",
        badge: "World-Famous",
        icon: "shield",
    },
    {
        name: "Tutankhamun",
        dynasty: "18th Dynasty",
        period: "New Kingdom",
        description:
            "Tutankhamun ruled Egypt during the 18th Dynasty (c. 1332–1323 BCE) and is famed for his nearly intact tomb discovered in 1922. Though his reign was short, his burial treasures—including the iconic gold mask—offer unparalleled insight into New Kingdom royal funerary practices.",
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
            "The Pyramids of Giza, built during Egypt’s Fourth Dynasty, form the last surviving wonder of the ancient world. These colossal tombs for Pharaohs Khufu, Khafre, and Menkaure display extraordinary engineering skill and remain symbols of Egypt’s power and spiritual belief in the afterlife.",
        badge: "Ancient Wonder",
        icon: "pyramid",
    },
    {
        name: "Sphinx",
        location: "Giza Plateau, Cairo",
        description:
            "The Great Sphinx of Giza is the most instantly recognizable statue associated with ancient Egypt and among the most famous in the world.",
        badge: "Iconic",
        icon: "landmark",
    },
    {
        name: "Temple of Karnak",
        location: "Luxor, Upper Egypt",
        description:
            "The Temple complex of Karnak, located on the east bank of Thebes, modern Luxor, in Upper Egypt, is considered the biggest temple in existence.",
        badge: "UNESCO",
        icon: "columns",
    },
    {
        name: "Temple of Luxor",
        location: "Luxor, Upper Egypt",
        description:
            "The Temple of Luxor, built by Pharaoh Amenhotep III and expanded by Ramesses II, stands majestically on the east bank of the Nile. Once the center of the Opet Festival, it symbolized the renewal of kingship and remains a stunning example of New Kingdom architecture.",
        badge: "Most Visited",
        icon: "compass",
    },
    {
        name: "The Great Temple of Ramesses II at Abu Simbel",
        location: "Aswan, Nubia",
        description:
            "Abu Simbel is a monumental rockcut complex situated on the western bank of the Nile. Built by the 19th Dynasty Pharaoh Ramesses II, who reigned from 1279–13 BCE, the complex consists of two temples carved into a sandstone cliff on the west bank of the Nile at the Second Nile Cataract, the border between Lower Nubia and Upper Nubia.",
        badge: "UNESCO",
        icon: "map-pin",
    },
];
