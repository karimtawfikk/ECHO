import chromadb
import math
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import time


print("hi")
qwen_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", tokenizer_kwargs={"padding_side": "left"})

test_queries_pharaohs = [
    ("Tell me about Khasekhemwy’s reign", "Khasekhem.txt"),
    ("Describe Djoser’s architectural achievements", "Djoser.txt"),
    ("Who ruled after Huni?", "Sneferu.txt"),
    ("Who built the Great Pyramid?", "Khufu.txt"),
    ("Tell me about the pharaoh who built the second pyramid of Giza", "Khafre.txt"),
    ("Describe the reign of Menkaure", "Menkaure.txt"),
    ("Who was the last king of the 4th dynasty?", "Shepseskaf.txt"),
    ("Tell me about the sun temples of Userkaf", "Userkaf.txt"),
    ("Who was Sahure's predecessor?", "Userkaf.txt"),
    ("Describe the reign of Niuserre", "Niuserre.txt"),
    ("Who was Pepi I?", "Pepi I.txt"),
    ("Tell me about the pyramid texts of Teti", "Teti.txt"),
    ("Who was the pharaoh Raneferef?", "Raneferef.txt"),
    ("Describe the reign of Khasekhem", "Khasekhem.txt"),
    ("Who was the pharaoh Sneferu?", "Sneferu.txt"),
    ("Who reunified Egypt in the Middle Kingdom?", "Mentuhotep II.txt"),
    ("Tell me about Amenemhat I", "Amenemhat I.txt"),
    ("Describe the military campaigns of Senwosret III", "Senwosret III.txt"),
    ("Who was the pharaoh Amenemhat III?", "Amenemhat III.txt"),
    ("Tell me about Senwosret I", "Senwosret I.txt"),
    ("Who was Senwosret IV?", "Senwosret IV.txt"),
    ("Describe the reign of Hor I", "Hor I.txt"),
    ("Who was Sobekhotep IV?", "Sobekhotep IV.txt"),
    ("Tell me about Sobekhotep V", "Sobekhotep V.txt"),
    ("Who was Sobekemsaf I?", "Sobekemsaf I.txt"),
    ("Tell me about the pharaoh Ahmose I", "Ahmose I.txt"),
    ("Who was the pharaoh Amenhotep II?", "Amenhotep II.txt"),
    ("Describe the reign of Thutmose III", "Thutmose III.txt"),
    ("Who was the mother of Thutmose III?", "Isis (mother of Thutmose III).txt"),
    ("Tell me about Thutmose IV", "Thutmose IV.txt"),
    ("Who was the 'Heretic Pharaoh'?", "Akhenaton.txt"),
    ("Tell me about Queen Nefertiti", "Nefertiti.txt"),
    ("Describe the reign of Tutankhamun", "Tutankhamun.txt"),
    ("Who was the pharaoh Horemheb?", "Horemheb.txt"),
    ("Tell me about Seti I's temple at Abydos", "Seti I.txt"),
    ("Describe the Battle of Kadesh and its leader", "Ramesses II.txt"),
    ("Who was the pharaoh Merenptah?", "Merenptah.txt"),
    ("Tell me about Ramesses III and the Sea Peoples", "Ramesess III.txt"),
    ("Who was Queen Hatshepsut?", "Hatshepsut.txt"),
    ("Describe Amenhotep III's building program", "Amenhotep III.txt"),
    ("Who was Tiye, wife of Amenhotep III?", "Tiye (Queen, wife of Amenhotep III).txt"),
    ("Who was the father of Queen Tiye?", "Yuya (father of Queen Tiye).txt"),
    ("Who was the mother of Queen Tiye?", "Thuya (mother of Queen Tiye).txt"),
    ("Tell me about Ankhsenamun", "Ankhsenamun.txt"),
    ("Who was Smenkhkare?", "Smenkhkare.txt"),
    ("Tell me about Seti II", "Seti II.txt"),
    ("Who was the pharaoh Meresankh?", "Meresankh (Queen, wife of Khafre).txt"),
    ("Describe the reign of Khamerernebty II", "Khamerernebty II.txt"),
    ("Who was Mutnofret?", "Mutnofret (Queen, Wife of Thutmose I and mother of Thutmose II).txt"),
    ("Tell me about Nofret", "Nofret (Queen, possibly Nofret II, wife of Senusret II).txt"),
    ("Who was Alexander the Great in Egypt?", "Alexander The Great.txt"),
    ("Describe the reign of Ptolemy I Soter", "Ptolemy I Soter.txt"),
    ("Who was Ptolemy II Philadelphus?", "Ptolemy II Philadelphus.txt"),
    ("Tell me about Ptolemy III Euergetes", "Ptolemy III Euergetes.txt"),
    ("Describe Cleopatra VII", "Cleopatra VII Philopator.txt"),
    ("Who was Nectanebo I?", "Nectanebo I.txt"),
    ("Who was Nectanebo II?", "Nectanebo II.txt"),
    ("Tell me about the reign of Achoris", "Achoris.txt"),
    ("Who was Amasis II?", "Amasis II.txt"),
    ("Describe the reign of Psusennes I", "Psusennes I.txt"),
    ("Who was Osorkon II?", "Osorkon II.txt"),
    ("Tell me about Neferkare Shabaka", "Neferkare Shabaka.txt"),
    ("Who was Amenirdis?", "Amenirdis (Daughter of Kashta).txt"),
    ("Describe Arsinoe II", "Arsinoe II (Queen, sister and wife of Ptolemy IV).txt"),
    ("Who was Arsinoe III?", "Arsinoe III (Queen, daughter of Ptolemy I and wife of Ptolemy II).txt"),
    ("Tell me about the god Amun", "Amun (God).txt"),
    ("Who is Amun-Ra?", "Amun-Ra (God).txt"),
    ("Describe the goddess Isis", "Isis (Goddess).txt"),
    ("Tell me about Osiris, god of the underworld", "Osiris (God).txt"),
    ("Who is the jackal-headed god Anubis?", "Anubis (God).txt"),
    ("Describe the goddess Hathor", "Hathor (Goddess).txt"),
    ("Who is the sun god Ra-Horakhty?", "Ra-Horakhty (God).txt"),
    ("Tell me about the god Horus", "Horus (God).txt"),
    ("Who is the creator god Ptah?", "Ptah (God).txt"),
    ("Describe the goddess Sekhmet", "Sekhmet (Goddess).txt"),

    #2nd
    ("Tell me about Khasekhemwy's reign", "Khasekhem.txt"),
    ("Describe Djoser's architectural achievements", "Djoser.txt"),
    ("Who ruled after Huni?", "Sneferu.txt"),
    ("What did Ahmose I accomplish?", "Ahmose I.txt"),
    ("Tell me about Akhenaton's religious revolution", "Akhenaton.txt"),
    ("Who was Hatshepsut and why was she unique?", "Hatshepsut.txt"),
    ("What is Khufu known for building?", "Khufu.txt"),
    ("Describe Ramesses II's military campaigns", "Ramesses II.txt"),
    ("Who was Tutankhamun and why is his tomb famous?", "Tutankhamun.txt"),
    ("What did Thutmose III achieve as a pharaoh?", "Thutmose III.txt"),
    ("Tell me about Amenhotep III's reign", "Amenhotep III.txt"),
    ("Who was Cleopatra VII Philopator?", "Cleopatra VII Philopator.txt"),
    ("What did Seti I build at Abydos?", "Seti I.txt"),
    ("Describe Amenemhat III's pyramid projects", "Amenemhat III.txt"),
    ("Who was Ptolemy I Soter?", "Ptolemy I Soter.txt"),
    ("Who was Sneferu and what pyramids did he build?", "Sneferu.txt"),
    ("Tell me about Khafre's reign and his pyramid", "Khafre.txt"),
    ("What is known about Menkaure?", "Menkaure.txt"),
    ("Describe the reign of Pepi I", "Pepi I.txt"),
    ("Who was Mentuhotep II and what did he achieve?", "Mentuhotep II.txt"),
    ("Tell me about Senwosret I's building projects", "Senwosret I.txt"),
    ("What did Senwosret III accomplish militarily?", "Senwosret III.txt"),
    ("Who was Amenemhat I and how did he come to power?", "Amenemhat I.txt"),
    ("Describe Amenhotep II's military campaigns", "Amenhotep II.txt"),
    ("Who was Merenptah and what is the Israel Stele?", "Merenptah.txt"),
    ("Tell me about Ramesses III's reign", "Ramesess III.txt"),
    ("What is known about Horemheb's rise to power?", "Horemheb.txt"),
    ("Who was Thutmose IV and what is the Dream Stele?", "Thutmose IV.txt"),
    ("Describe the reign of Userkaf", "Userkaf.txt"),
    ("Who was Niuserre and what did he build?", "Niuserre.txt"),
    ("Tell me about Shepseskaf's reign", "Shepseskaf.txt"),
    ("What is known about Teti's rule?", "Teti.txt"),
    ("Who was Raneferef?", "Raneferef.txt"),
    ("Describe Sobekhotep IV's reign", "Sobekhotep IV.txt"),
    ("Who was Sobekhotep V?", "Sobekhotep V.txt"),
    ("Tell me about Sobekemsaf I", "Sobekemsaf I.txt"),
    ("Who was Neferkare Shabaka?", "Neferkare Shabaka.txt"),
    ("What is known about Osorkon II?", "Osorkon II.txt"),
    ("Describe the reign of Psusennes I", "Psusennes I.txt"),
    ("Who was Nectanebo I and what did he build?", "Nectanebo I.txt"),
    ("Tell me about Nectanebo II, the last native pharaoh", "Nectanebo II.txt"),
    ("Who was Amasis II?", "Amasis II.txt"),
    ("What is known about Achoris?", "Achoris.txt"),
    ("Describe Alexander The Great's role in Egypt", "Alexander The Great.txt"),
    ("Who was Ptolemy II Philadelphus?", "Ptolemy II Philadelphus.txt"),
    ("Tell me about Ptolemy III Euergetes", "Ptolemy III Euergetes.txt"),
    ("Who was Seti II?", "Seti II.txt"),
    ("What is known about Smenkhkare?", "Smenkhkare.txt"),
    ("Describe the reign of Hor I", "Hor I.txt"),
    ("Who was Senwosret IV?", "Senwosret IV.txt"),
    ("Who was the god Amun in Egyptian mythology?", "Amun (God).txt"),
    ("Describe the role of Amun-Ra in Egyptian religion", "Amun-Ra (God).txt"),
    ("Who was the goddess Hathor?", "Hathor (Goddess).txt"),
    ("Tell me about the goddess Isis", "Isis (Goddess).txt"),
    ("Who was Horus and what is his significance?", "Horus (God).txt"),
    ("Describe the god Osiris and the afterlife", "Osiris (God).txt"),
    ("Who was the goddess Sekhmet?", "Sekhmet (Goddess).txt"),
    ("Tell me about the god Ptah", "Ptah (God).txt"),
    ("Who was the goddess Mut?", "Mut (Goddess).txt"),
    ("Describe the god Anubis and his role", "Anubis (God).txt"),
    ("Who was the goddess Taweret?", "Taweret (Goddess).txt"),
    ("Tell me about the god Khonsu", "Khonsu (God).txt"),
    ("Who was the goddess Anath?", "Anath Goddess.txt"),
    ("Describe the god Seth in Egyptian mythology", "Seth (God).txt"),
    ("Who was the god Serapis?", "Serapis (God).txt"),
    ("Tell me about Ra-Horakhty", "Ra-Horakhty (God).txt"),
    ("Who was the goddess Bat?", "Bat Goddess.txt"),
    ("Describe the god Hauron", "Hauron God.txt"),
    ("Who was Nefertiti and what was her role?", "Nefertiti.txt"),
    ("Tell me about Queen Tiye, wife of Amenhotep III", "Tiye (Queen, wife of Amenhotep III).txt"),
    ("Who was Ankhsenamun?", "Ankhsenamun.txt"),
    ("Describe Queen Meresankh, wife of Khafre", "Meresankh (Queen, wife of Khafre).txt"),
    ("Who was Khamerernebty II?", "Khamerernebty II.txt"),
    ("Tell me about Mutnofret, wife of Thutmose I", "Mutnofret (Queen, Wife of Thutmose I and mother of Thutmose II).txt"),
    ("Who was Amenirdis, daughter of Kashta?", "Amenirdis (Daughter of Kashta).txt"),
    ("Who was Yuya, father of Queen Tiye?", "Yuya (father of Queen Tiye).txt"),
    ("Describe Thuya, mother of Queen Tiye", "Thuya (mother of Queen Tiye).txt"),
    ("Who was Isis, mother of Thutmose III?", "Isis (mother of Thutmose III).txt"),
    ("Tell me about Arsinoe II, sister and wife of Ptolemy IV", "Arsinoe II (Queen, sister and wife of Ptolemy IV).txt"),
    ("Describe Arsinoe III, daughter of Ptolemy I", "Arsinoe III (Queen, daughter of Ptolemy I and wife of Ptolemy II).txt"),
    ("Who was Nofret, wife of Senusret II?", "Nofret (Queen, possibly Nofret II, wife of Senusret II).txt"),
    #3rd:
    ("Who was the mother of the king who built the smallest Giza pyramid?", "Mutnofret (Queen, Wife of Thutmose I and mother of Thutmose II).txt"),
    ("Which pharaoh's father was Yuya and whose daughter was Ankhsenamun?", "Tiye (Queen, wife of Amenhotep III).txt"),
    ("Who was the biological mother of the pharaoh who restored the old gods after the Amarna period?", "Isis (mother of Thutmose III).txt"),
    ("Identify the ruler who was the son of Sneferu and the father of Khafre.", "Khufu.txt"),
    ("Who was the female coregent believed to have ruled briefly after Akhenaten?", "Nefertiti.txt"),
    ("Which pharaoh was the grandson of Amenhotep II and father of Akhenaten?", "Amenhotep III.txt"),
    ("Who was the wife of the king who commissioned the Great Sphinx?", "Meresankh (Queen, wife of Khafre).txt"),
    ("Which pharaoh of the 18th dynasty was the stepson of Hatshepsut?", "Thutmose III.txt"),
    ("Identify the queen who was both the sister and wife of Ptolemy IV.", "Arsinoe II (Queen, sister and wife of Ptolemy IV).txt"),
    ("Who was the father of the queen who famously committed suicide after the Battle of Actium?", "Ptolemy I Soter.txt"),
    ("Which king is depicted on the Narmer Palette's contemporary, the Scorpion King?", "Khasekhem.txt"),
    ("Who was the last native Egyptian pharaoh before the second Persian conquest?", "Nectanebo II.txt"),
    ("Identify the ruler who survived a harem conspiracy only to be succeeded by Ramesses IV.", "Ramesess III.txt"),
    ("Which pharaoh's name means 'The one who brings Maat back'?", "Tutankhamun.txt"),
    ("Who was the king that moved the capital to Itj-tawy?", "Amenemhat I.txt"),
    ("Identify the god-king whose cult was centered at Abydos as the Lord of the Dead.", "Osiris (God).txt"),
    ("Which pharaoh is associated with the 'Famine Stele' found on Sehel Island?", "Djoser.txt"),
    ("Who was the king that led seventeen military campaigns into the Levant?", "Thutmose III.txt"),
    ("Identify the pharaoh who signed the first known peace treaty with the Hittites.", "Ramesses II.txt"),
    ("Who was the first king to record the use of 'Pyramid Texts' in his tomb?", "Unas.txt"),
]

test_queries_landmarks = [
    ("Who built the Step Pyramid?", "Pyramid of Saqqara - Pyramid of Djoser.txt"),
    ("Describe the Great Sphinx construction", "Sphinx.txt"),
    ("Which Pharaoh commissioned Karnak Temple?", "Temple of Karnak.txt"),
    ("Tell me about the Pyramids of Giza", "Pyramids of Giza.txt"),
    ("Describe the Sphinx of Memphis", "Sphinx of Memphis.txt"),
    ("Who built the Pyramid of Unas?", "Pyramid of Unas.txt"),
    ("Tell me about the Pyramid of Userkaf", "Pyramid of Userkaf.txt"),
    ("Describe the Tomb of Horemheb in Saqqara", "Tomb of Horemheb in Saqqara.txt"),
    ("Tell me about the Pyramid of Sahure", "Pyramid of Sahure.txt"),
    ("Describe the Pyramid of Meidum", "Pyramid of Meidum.txt"),
    ("Who built the Bent Pyramid?", "Pyramid of Senefru - Bent Pyramid of Senefru.txt"),
    ("Tell me about the Pyramid of Seila", "Pyramid of Seila.txt"),
    ("Describe the Hawara Pyramid", "Pyramid of Hawara - White pyramid of Amnemhat III.txt"),
    ("What is the Black Pyramid?", "Black Pyramid of Amenemhat III.txt"),
    ("Tell me about the Serapeum of Alexandria", "Pompeys Pillar - Serapeum of Alexandria.txt"),
    ("Describe the Temple of Luxor", "Temple of Luxor.txt"),
    ("Tell me about the Karnak Temple complex", "Temple of Karnak.txt"),
    ("Who built the Temple of Khonsu?", "Temple of Khonsu in Karnak.txt"),
    ("Describe the Temple of Hatshepsut", "Temple of Hatshepsut in Deir El Bahari.txt"),
    ("What are the Colossi of Memnon?", "Colossoi of Memnon.txt"),
    ("Describe the Ramesseum", "Ramessum.txt"),
    ("Tell me about the Temple of Medinet Habu", "Temple of Habu.txt"),
    ("Who is honored at the Temple of Hathor in Deir el Medina?", "Ptolemaic Temple of Hathor in Deir el Medina.txt"),
    ("Describe the Temple of Isis in Deir Shelwit", "Temple of Isis in Deir Shelwit.txt"),
    ("Tell me about the mountain Al Qurn", "Al Qurn.txt"),
    ("Describe the Temple of Esna", "Temple of Esna.txt"),
    ("Tell me about the Temple of Edfu", "Temple of Horus at Edfu.txt"),
    ("Describe the Temple of Kom Ombo", "Temple of Kom Ombo.txt"),
    ("Who is the Temple of Hathor in Dendera for?", "Temple of Hathor in Dendera.txt"),
    ("Describe the Roman Mammisi in Dendera", "The Roman Mammisi in Dendera.txt"),
    ("Tell me about the Osireion at Abydos", "Osireion.txt"),
    ("Describe the Temple of Seti I in Abydos", "Temple of Seti I in Abydos.txt"),
    ("Tell me about the Temple of Nekhbet and Hathor at El Kab", "Temple of Nekhbet and Hathor at El Kab.txt"),
    ("Who is Merit Amun's temple for?", "Temple of Merit Amun.txt"),
    ("Tell me about the Speos of Horemheb", "Speos of Horemheb.txt"),
    ("Tell me about the Great Temple of Abu Simbel", "The Great Temple of Ramesses II at Abu Simbel.txt"),
    ("Describe the Small Temple of Abu Simbel", "Small Temple of Hathor and Nefertari at Abu Simbel.txt"),
    ("Tell me about the Temple of Isis in Philae", "Temple of Isis in Philae.txt"),
    ("Describe the Kiosk of Trajan", "Kiosk of Trajan in Philae.txt"),
    ("Tell me about the Temple of Hathor in Philae", "Temple of Hathor in Philae.txt"),
    ("Describe the Temple of Kalabsha", "Temple of Kalabsha.txt"),
    ("Tell me about the Temple of Amada", "Temple of Amada.txt"),
    ("Describe the Temple of Dakka", "Temple of Dakka.txt"),
    ("Who is honored at the Temple of Derr?", "Temple of Derr.txt"),
    ("Tell me about the Temple of Wadi es-Sebua", "Temple of Wadi es-Sebua - Temple of Ramesses II.txt"),
    ("Describe the Temple of Maharraqa", "Temple of Maharraqa.txt"),
    ("What is the Kiosk of Qertassi?", "Kiosk of Qertassi.txt"),
    ("Tell me about the Famine Stele", "Famine Stele.txt"),
    ("Describe the Temple of Hibis", "Temple of Hibis.txt"),
    ("Tell me about Qasr Qarun", "Qasr Qarun.txt"),
    ("Describe the Tomb of Petosiris", "Tomb of Petosiris.txt"),
    ("Tell me about the Tomb of Isadora", "Tomb of Isadora.txt"),
    ("What is the Small Aten Temple?", "The Small Aten Temple.txt"),
    ("Tell me about Pompey's Pillar", "Pompeys Pillar - Serapeum of Alexandria.txt"),
    ("Describe the Temple of Horus at Edfu", "Temple of Horus at Edfu.txt"),
    ("Tell me about the Temple of Karnak architecture", "Temple of Karnak.txt"),
    ("Describe the pylons of Luxor Temple", "Temple of Luxor.txt"),
    ("Tell me about the hypostyle hall at Karnak", "Temple of Karnak.txt"),
    ("Describe the statues at Abu Simbel", "The Great Temple of Ramesses II at Abu Simbel.txt"),
    ("Tell me about the reliefs in Dendera", "Temple of Hathor in Dendera.txt"),
    ("Describe the location of Philae Temple", "Temple of Isis in Philae.txt"),
    ("Tell me about the sanctuary at Kom Ombo", "Temple of Kom Ombo.txt"),
    ("Describe the pyramid field at Saqqara", "Pyramid of Saqqara - Pyramid of Djoser.txt"),
    ("Tell me about the Sphinx's dream stele", "Sphinx.txt"),
    ("Describe the Valley of the Queens context for Deir Shelwit", "Temple of Isis in Deir Shelwit.txt"),
    ("Tell me about the astronomical ceiling in Dendera", "Temple of Hathor in Dendera.txt"),
    ("Describe the healing statues in the Serapeum", "Pompeys Pillar - Serapeum of Alexandria.txt"),
    ("Tell me about the sacred lake at Karnak", "Temple of Karnak.txt"),
    #2nd:
    ("Who built the Step Pyramid?", "Pyramid of Saqqara - Pyramid of Djoser.txt"),
    ("Describe the Great Sphinx construction", "Sphinx.txt"),
    ("Which Pharaoh commissioned Karnak Temple?", "Temple of Karnak.txt"),
    ("What are the Pyramids of Giza and who built them?", "Pyramids of Giza.txt"),
    ("Tell me about the Temple of Hatshepsut in Deir El Bahari", "Temple of Hatshepsut in Deir El Bahari.txt"),
    ("Describe the Great Temple of Ramesses II at Abu Simbel", "The Great Temple of Ramesses II at Abu Simbel.txt"),
    ("What is special about the Temple of Horus at Edfu?", "Temple of Horus at Edfu.txt"),
    ("Tell me about the Temple of Hathor in Dendera", "Temple of Hathor in Dendera.txt"),
    ("What is the Osireion and where is it located?", "Osireion.txt"),
    ("Describe the Temple of Luxor", "Temple of Luxor.txt"),
    ("What deities are worshipped at the Temple of Kom Ombo?", "Temple of Kom Ombo.txt"),
    ("Tell me about the Small Temple of Hathor and Nefertari at Abu Simbel", "Small Temple of Hathor and Nefertari at Abu Simbel.txt"),
    ("What is the Famine Stele?", "Famine Stele.txt"),
    ("Describe the Colossoi of Memnon", "Colossoi of Memnon.txt"),
    ("What is the Kiosk of Trajan in Philae?", "Kiosk of Trajan in Philae.txt"),
    ("Tell me about the Tomb of Horemheb in Saqqara", "Tomb of Horemheb in Saqqara.txt"),
    ("What is Al Qurn and its significance?", "Al Qurn.txt"),
    ("Describe the Black Pyramid of Amenemhat III", "Black Pyramid of Amenemhat III.txt"),
    ("Tell me about the Kiosk of Qertassi", "Kiosk of Qertassi.txt"),
    ("What is Pompey's Pillar and the Serapeum of Alexandria?", "Pompeys Pillar - Serapeum of Alexandria.txt"),
    ("Describe the Ptolemaic Temple of Hathor in Deir el Medina", "Ptolemaic Temple of Hathor in Deir el Medina.txt"),
    ("Tell me about the Pyramid of Hawara", "Pyramid of Hawara - White pyramid of Amnemhat III.txt"),
    ("What is the Pyramid of Meidum and its history?", "Pyramid of Meidum.txt"),
    ("Describe the Pyramid of Sahure", "Pyramid of Sahure.txt"),
    ("Tell me about the Pyramid of Seila", "Pyramid of Seila.txt"),
    ("What is the Bent Pyramid of Sneferu?", "Pyramid of Senefru - Bent Pyramid of Senefru.txt"),
    ("Describe the Pyramid of Unas and its texts", "Pyramid of Unas.txt"),
    ("Tell me about the Pyramid of Userkaf", "Pyramid of Userkaf.txt"),
    ("What is Qasr Qarun?", "Qasr Qarun.txt"),
    ("Describe the Ramesseum temple complex", "Ramessum.txt"),
    ("Tell me about the Speos of Horemheb", "Speos of Horemheb.txt"),
    ("What is the Sphinx of Memphis?", "Sphinx of Memphis.txt"),
    ("Describe the Temple of Amada", "Temple of Amada.txt"),
    ("Tell me about the Temple of Dakka", "Temple of Dakka.txt"),
    ("What is the Temple of Derr?", "Temple of Derr.txt"),
    ("Describe the Temple of Esna", "Temple of Esna.txt"),
    ("Tell me about the Temple of Habu", "Temple of Habu.txt"),
    ("What is the Temple of Hathor in Philae?", "Temple of Hathor in Philae.txt"),
    ("Describe the Temple of Hibis", "Temple of Hibis.txt"),
    ("Tell me about the Temple of Isis in Deir Shelwit", "Temple of Isis in Deir Shelwit.txt"),
    ("What is the Temple of Isis in Philae?", "Temple of Isis in Philae.txt"),
    ("Describe the Temple of Kalabsha", "Temple of Kalabsha.txt"),
    ("Tell me about the Temple of Khonsu in Karnak", "Temple of Khonsu in Karnak.txt"),
    ("What is the Temple of Maharraqa?", "Temple of Maharraqa.txt"),
    ("Describe the Temple of Merit Amun", "Temple of Merit Amun.txt"),
    ("Tell me about the Temple of Nekhbet and Hathor at El Kab", "Temple of Nekhbet and Hathor at El Kab.txt"),
    ("What is the Temple of Seti I in Abydos?", "Temple of Seti I in Abydos.txt"),
    ("Describe the Temple of Wadi es-Sebua", "Temple of Wadi es-Sebua - Temple of Ramesses II.txt"),
    ("Tell me about The Roman Mammisi in Dendera", "The Roman Mammisi in Dendera.txt"),
    ("What is The Small Aten Temple?", "The Small Aten Temple.txt"),
    ("Describe the Tomb of Isadora", "Tomb of Isadora.txt"),
    ("Tell me about the Tomb of Petosiris", "Tomb of Petosiris.txt"),
    ("Who commissioned the Pyramids of Giza and how were they built?", "Pyramids of Giza.txt"),
    ("What architectural innovations are found in the Step Pyramid?", "Pyramid of Saqqara - Pyramid of Djoser.txt"),
    ("Describe the religious significance of the Temple of Karnak", "Temple of Karnak.txt"),
    ("What makes the Temple of Hatshepsut unique architecturally?", "Temple of Hatshepsut in Deir El Bahari.txt"),
    ("Tell me about the relocation of Abu Simbel temples", "The Great Temple of Ramesses II at Abu Simbel.txt"),
    ("What deities are honored at the Temple of Edfu?", "Temple of Horus at Edfu.txt"),
    ("Describe the astronomical features of Dendera Temple", "Temple of Hathor in Dendera.txt"),
    ("What is the purpose of the Osireion at Abydos?", "Osireion.txt"),
    ("Tell me about the sphinxes at the Temple of Luxor", "Temple of Luxor.txt"),
    ("What is unique about the dual design of Kom Ombo Temple?", "Temple of Kom Ombo.txt"),
    ("Describe the statues in the Small Temple at Abu Simbel", "Small Temple of Hathor and Nefertari at Abu Simbel.txt"),
    ("What does the Famine Stele tell us about ancient Egypt?", "Famine Stele.txt"),
    ("Who do the Colossi of Memnon represent?", "Colossoi of Memnon.txt"),
    ("What is the architectural style of the Kiosk of Trajan?", "Kiosk of Trajan in Philae.txt"),
    ("Tell me about the tomb decorations in Horemheb's Saqqara tomb", "Tomb of Horemheb in Saqqara.txt"),
    ("What is the significance of Al Qurn mountain?", "Al Qurn.txt"),
    ("Describe the construction challenges of the Black Pyramid", "Black Pyramid of Amenemhat III.txt"),
    ("What rituals were performed at the Kiosk of Qertassi?", "Kiosk of Qertassi.txt"),
    ("Tell me about the Serapeum's role in Alexandria", "Pompeys Pillar - Serapeum of Alexandria.txt"),
    ("What is the Bent Pyramid's unusual angle and why?", "Pyramid of Senefru - Bent Pyramid of Senefru.txt"),
    ("Describe the Pyramid Texts found in Unas's pyramid", "Pyramid of Unas.txt"),
    ("What is the historical importance of the Ramesseum?", "Ramessum.txt"),
    ("Tell me about the Temple of Seti I's reliefs at Abydos", "Temple of Seti I in Abydos.txt"),
    #3rd:
    ("Which temple was completely dismantled and moved to Agilkia Island to avoid flooding?", "Temple of Isis in Philae.txt"),
    ("Identify the structure where the sun illuminates the inner sanctuary only twice a year.", "The Great Temple of Ramesses II at Abu Simbel.txt"),
    ("Which pyramid changed its slope halfway through construction due to structural instability?", "Pyramid of Senefru - Bent Pyramid of Senefru.txt"),
    ("Identify the 'Labyrinth' of Egypt mentioned by Herodotus.", "Pyramid of Hawara - White pyramid of Amnemhat III.txt"),
    ("Which monument was built by a queen to justify her right to rule through divine birth?", "Temple of Hatshepsut in Deir El Bahari.txt"),
    ("Identify the Greco-Roman temple that is perfectly symmetrical and shared by two gods.", "Temple of Kom Ombo.txt"),
    ("Which location is famous for the 'Dream Stele' placed between its paws?", "Sphinx.txt"),
    ("Identify the temple that contains a famous relief of an ancient 'light bulb' or 'zodiac' on its ceiling.", "Temple of Hathor in Dendera.txt"),
    ("Which site is the only known temple dedicated to the god of the oasis, Amun-Hibis?", "Temple of Hibis.txt"),
    ("Identify the mudbrick pyramid that has almost completely eroded into a black mound.", "Black Pyramid of Amenemhat III.txt"),
    ("Which temple was the main center for the cult of the falcon-headed god Horus?", "Temple of Horus at Edfu.txt"),
    ("Identify the massive statues that 'sang' at dawn according to Greek travelers.", "Colossoi of Memnon.txt"),
    ("Which temple complex is connected to the Nile via a canal and houses the Great Hypostyle Hall?", "Temple of Karnak.txt"),
    ("Identify the underground gallery at Saqqara where the sacred Apis bulls were buried.", "Pyramid of Saqqara - Pyramid of Djoser.txt"),
    ("Which temple served as the 'Southern Harem' for the god Amun?", "Temple of Luxor.txt"),
    ("Identify the tomb chapel famous for its 'false door' and daily offering scenes of the official Menna.", "Tomb of Horemheb in Saqqara.txt"),
    ("Which structure is often mistaken for a fortress due to its high mudbrick walls in Medinet Habu?", "Temple of Habu.txt"),
    ("Identify the small temple dedicated to the goddess of love, located near the Great Temple of Ramesses.", "Small Temple of Hathor and Nefertari at Abu Simbel.txt"),
    ("Which site features a Roman-era pillar that was wrongly named after a general of Julius Caesar?", "Pompeys Pillar - Serapeum of Alexandria.txt"),
    ("Identify the temple located inside the 'Hill of the Horn' that overlooks the Valley of the Kings.", "Al Qurn.txt"),
]


def evaluate_model(persist_path, collection_name, model, queries, k=5, mrl_dim=None):

    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_collection(collection_name)

    recall_at_1 = 0
    recall_at_k = 0
    mrr_total = 0
    ndcg_total = 0

    total_search_time = 0

    for query_text, ground_truth_file in queries:

        # --- Embed (not timing this) ---
        query_embedding = model.encode([query_text], normalize_embeddings=True)

        if mrl_dim is not None:
            query_embedding = query_embedding[:, :mrl_dim]
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # --- Measure search time only ---
        start = time.perf_counter()
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=1
        )
        end = time.perf_counter()

        total_search_time += (end - start)

        retrieved = results["metadatas"][0]

        dcg = 0
        for rank, meta in enumerate(retrieved):
            if meta["entity_name"] == ground_truth_file:
                recall_at_k += 1
                if rank == 0:
                    recall_at_1 += 1
                mrr_total += 1 / (rank + 1)
                dcg = 1 / math.log2(rank + 2)
                break

        ndcg_total += dcg

    total = len(queries)

    return {
        "Recall@1": recall_at_1 / total,
        f"Recall@{k}": recall_at_k / total,
        "MRR": mrr_total / total,
        f"NDCG@{k}": ndcg_total / total,
        "Avg Search Time (s)": total_search_time / total
    }


def average_metrics(metrics1, metrics2):
    avg = {}
    for key in metrics1.keys():
        avg[key] = (metrics1[key] + metrics2[key]) / 2
    return avg



#Pharaohs 
#Qwen Evaluation:
results_pharaohs_qwen = evaluate_model(
    Path(r"C:\Uni\4th Year\GP\ECHO\data\chatbot\embeddings\pharaohs_qwen_db"),
    "pharaohs",
    qwen_model,
    test_queries_pharaohs,
    k=5
)

#Pharaohs 
#MRL Qwen Evaluation:
results_pharaohs_qwen_mrl_512 = evaluate_model(
    Path(r"C:\Uni\4th Year\GP\ECHO\data\chatbot\embeddings\pharaohs_qwen_MRL_512_db"),
    "pharaohs",
    qwen_model,
    test_queries_pharaohs,
    k=5,
    mrl_dim=512
)

results_pharaohs_qwen_mrl_768 = evaluate_model(
    Path(r"C:\Uni\4th Year\GP\ECHO\data\chatbot\embeddings\pharaohs_qwen_MRL_768_db"),
    "pharaohs",
    qwen_model,
    test_queries_pharaohs,
    k=5,
    mrl_dim=768
)



print("PHARAOHS - Qwen:", results_pharaohs_qwen)
print("PHARAOHS - QWEN 512:", results_pharaohs_qwen_mrl_512)
print("PHARAOHS - QWEN 768:", results_pharaohs_qwen_mrl_768)


#Landmarks

#Qwen Evaluation:
results_landmarks_qwen = evaluate_model(
    Path(r"C:\Uni\4th Year\GP\ECHO\data\chatbot\embeddings\landmarks_qwen_db"),
    "landmarks",
    qwen_model,
    test_queries_landmarks,
    k=5
)

#MRL Qwen Evaluation:
results_landmarks_qwen_mrl_512 = evaluate_model(
    Path(r"C:\Uni\4th Year\GP\ECHO\data\chatbot\embeddings\landmarks_qwen_MRL_512_db"),
    "landmarks",
    qwen_model,
    test_queries_landmarks,
    k=5,
    mrl_dim=512
)

results_landmarks_qwen_mrl_768 = evaluate_model(
    Path(r"C:\Uni\4th Year\GP\ECHO\data\chatbot\embeddings\landmarks_qwen_MRL_768_db"),
    "landmarks",
    qwen_model,
    test_queries_landmarks,
    k=5,
    mrl_dim=768
)



print("LANDMARKS - Qwen:", results_landmarks_qwen)
print("LANDMARKS - Qwen 512:", results_landmarks_qwen_mrl_512)
print("LANDMARKS - Qwen 768:", results_landmarks_qwen_mrl_768)



overall_qwen = average_metrics(results_pharaohs_qwen, results_landmarks_qwen)
overall_qwen_mrl_512 = average_metrics(results_pharaohs_qwen_mrl_512, results_landmarks_qwen_mrl_512)
overall_qwen_mrl_768 = average_metrics(results_pharaohs_qwen_mrl_768, results_landmarks_qwen_mrl_768)


print("Qwen overall:", overall_qwen)
print("Qwen 512 overall:", overall_qwen_mrl_512)
print("Qwen 768 overall:", overall_qwen_mrl_768)