"""Drug Database & Formulary Engine.
Provides comprehensive brand & generic drug lookup, fuzzy string matching,
phonetic normalization, standard dosages, and therapeutic class mapping.
"""

from typing import Dict, List, Optional, Tuple, Any
import re
from rapidfuzz import fuzz, process

# Curated reference database of common prescription medications
DRUG_DATABASE: List[Dict[str, Any]] = [
    {
        "id": "DRUG_001",
        "generic_name": "Paracetamol",
        "synonyms": ["Acetaminophen", "APAP", "Paracet", "PCM", "Para"],
        "brands": ["Dolo", "Dolo 650", "Calpol", "Crocin", "Panadol", "Tylenol", "Pacimol"],
        "class": "Analgesic / Antipyretic",
        "typical_doses": ["500mg", "650mg", "1000mg", "250mg/5ml"],
        "max_single_dose_mg": 1000,
        "max_daily_dose_mg": 4000,
        "route": "Oral / IV",
        "description": "Used to treat mild to moderate pain and fever."
    },
    {
        "id": "DRUG_002",
        "generic_name": "Amoxicillin",
        "synonyms": ["Amoxycillin", "Amox", "Amoxi", "Novamox", "Novamax", "Novamox 500", "Novamax 500", "Novamox 500mg", "Novamax 500mg", "Xonon"],
        "brands": ["Amoxil", "Novamox", "Novamox 500mg", "Novamax", "Mox", "Moxikind", "Almox"],
        "class": "Aminopenicillin Antibiotic",
        "typical_doses": ["500mg", "250mg", "875mg"],
        "max_single_dose_mg": 1000,
        "max_daily_dose_mg": 3000,
        "route": "Oral",
        "description": "Broad-spectrum penicillin antibiotic for bacterial infections."
    },
    {
        "id": "DRUG_003",
        "generic_name": "Amoxicillin + Clavulanic Acid",
        "synonyms": ["Co-amoxiclav", "Amox-Clav", "Augmentin", "Clavam", "Azigmentun", "Augmentui", "Argmentun"],
        "brands": ["Augmentin", "Augmentin 625", "Augmentin 625mg", "Clavam 625", "Moxikind-CV", "Megamentin", "Azigmentun", "Augmentui", "Argmentun"],
        "class": "Beta-lactam Antibiotic + Inhibitor",
        "typical_doses": ["625mg", "375mg", "1000mg"],
        "max_single_dose_mg": 1000,
        "max_daily_dose_mg": 2000,
        "route": "Oral / IV",
        "description": "Potent broad-spectrum antibiotic for resistant respiratory, skin, and ENT infections."
    },
    {
        "id": "DRUG_004",
        "generic_name": "Ibuprofen",
        "synonyms": ["Ibu", "Ibup"],
        "brands": ["Brufen", "Advil", "Motrin", "Ibugesic"],
        "class": "NSAID",
        "typical_doses": ["200mg", "400mg", "600mg", "800mg"],
        "max_single_dose_mg": 800,
        "max_daily_dose_mg": 2400,
        "route": "Oral",
        "description": "Nonsteroidal anti-inflammatory drug for pain, inflammation, and fever."
    },
    {
        "id": "DRUG_005",
        "generic_name": "Metformin",
        "synonyms": ["Metform", "Metfor", "Glycomet", "Glycomet SR", "Guyemet", "Guyemet sr", "Glycomet SR 500", "Gryeomet", "Glycomet 500"],
        "brands": ["Glycomet SR", "Glycomet", "Glucophage", "Obimet", "Cetapin", "Fortamet"],
        "class": "Biguanide Antidiabetic",
        "typical_doses": ["500mg", "850mg", "1000mg"],
        "max_single_dose_mg": 1000,
        "max_daily_dose_mg": 2550,
        "route": "Oral",
        "description": "First-line oral medication for Type 2 diabetes."
    },
    {
        "id": "DRUG_006",
        "generic_name": "Omeprazole",
        "synonyms": ["Omep", "Omepraz"],
        "brands": ["Prilosec", "Omez", "Losec", "Ocid", "Omee"],
        "class": "Proton Pump Inhibitor (PPI)",
        "typical_doses": ["20mg", "40mg"],
        "max_single_dose_mg": 40,
        "max_daily_dose_mg": 80,
        "route": "Oral",
        "description": "Reduces stomach acid for GERD, gastritis, and peptic ulcers."
    },
    {
        "id": "DRUG_007",
        "generic_name": "Pantoprazole",
        "synonyms": ["Panto", "Pantop", "Pantop 40", "Pantup", "Pan 40"],
        "brands": ["Pantop 40", "Pantop", "Pantocid", "Protonix", "Pan 40", "Pantodac", "Pan-D component"],
        "class": "Proton Pump Inhibitor (PPI)",
        "typical_doses": ["40mg", "20mg"],
        "max_single_dose_mg": 40,
        "max_daily_dose_mg": 80,
        "route": "Oral / IV",
        "description": "Suppresses gastric acid secretion for acid reflux and ulcer healing."
    },
    {
        "id": "DRUG_008",
        "generic_name": "Atorvastatin",
        "synonyms": ["Atorva", "Ator", "Avos", "Avae", "AAvo", "Ator 10", "Avos 10"],
        "brands": ["Ator", "Avos", "Lipitor", "Atorva", "Storvas", "Atocor", "Tonact"],
        "class": "HMG-CoA Reductase Inhibitor (Statin)",
        "typical_doses": ["10mg", "20mg", "40mg", "80mg"],
        "max_single_dose_mg": 80,
        "max_daily_dose_mg": 80,
        "route": "Oral",
        "description": "Lowers LDL cholesterol and cardiovascular disease risk."
    },
    {
        "id": "DRUG_009",
        "generic_name": "Azithromycin",
        "synonyms": ["Azithro", "Azith", "Azithral"],
        "brands": ["Zithromax", "Azee", "Azithral", "Zady", "Azax"],
        "class": "Macrolide Antibiotic",
        "typical_doses": ["250mg", "500mg"],
        "max_single_dose_mg": 500,
        "max_daily_dose_mg": 500,
        "route": "Oral",
        "description": "Macrolide antibiotic for respiratory tract and soft tissue infections."
    },
    {
        "id": "DRUG_010",
        "generic_name": "Cetirizine",
        "synonyms": ["Cetzine", "Cetriz"],
        "brands": ["Zyrtec", "Cetzine", "Alerid", "Okacet", "Incid-L"],
        "class": "Second-generation Antihistamine",
        "typical_doses": ["5mg", "10mg"],
        "max_single_dose_mg": 10,
        "max_daily_dose_mg": 10,
        "route": "Oral",
        "description": "Relieves allergic rhinitis, urticaria, and sneezing."
    },
    {
        "id": "DRUG_011",
        "generic_name": "Levocetirizine",
        "synonyms": ["L-Cetirizine", "Levocet", "Lejet", "Safecet", "Lejt", "Safcet"],
        "brands": ["Xyzal", "Levocet", "1-AL", "Vozet", "Teczine", "Lejet", "Safecet", "Lejet 5", "Safecet 5", "Lejt", "Safcet"],
        "class": "Antihistamine",
        "typical_doses": ["2.5mg", "5mg"],
        "max_single_dose_mg": 5,
        "max_daily_dose_mg": 5,
        "route": "Oral",
        "description": "Non-sedating antihistamine for allergic allergies and hives."
    },
    {
        "id": "DRUG_012",
        "generic_name": "Montelukast",
        "synonyms": ["Montelu", "Mont"],
        "brands": ["Singulair", "Montair", "Montek", "Romilast"],
        "class": "Leukotriene Receptor Antagonist",
        "typical_doses": ["4mg", "5mg", "10mg"],
        "max_single_dose_mg": 10,
        "max_daily_dose_mg": 10,
        "route": "Oral",
        "description": "Prevents asthma attacks and treats chronic allergic rhinitis."
    },
    {
        "id": "DRUG_013",
        "generic_name": "Ciprofloxacin",
        "synonyms": ["Cipro", "Ciproflox"],
        "brands": ["Cipro", "Ciplox", "Cifran", "Ciprobid"],
        "class": "Fluoroquinolone Antibiotic",
        "typical_doses": ["250mg", "500mg", "750mg"],
        "max_single_dose_mg": 750,
        "max_daily_dose_mg": 1500,
        "route": "Oral / IV / Ophthalmic",
        "description": "Broad-spectrum antibacterial for urinary and abdominal infections."
    },
    {
        "id": "DRUG_014",
        "generic_name": "Amlodipine",
        "synonyms": ["Amlo", "Amlodac", "Amloalcc", "Amlodae", "Amlodac 5"],
        "brands": ["Amlodac", "Amlong", "Norvasc", "Amlopin", "Amcard"],
        "class": "Dihydropyridine Calcium Channel Blocker",
        "typical_doses": ["2.5mg", "5mg", "10mg"],
        "max_single_dose_mg": 10,
        "max_daily_dose_mg": 10,
        "route": "Oral",
        "description": "Used to treat hypertension and chronic stable angina."
    },
    {
        "id": "DRUG_015",
        "generic_name": "Telmisartan",
        "synonyms": ["Telmi", "Telm"],
        "brands": ["Micardis", "Telma", "Telmikind", "Cresar"],
        "class": "Angiotensin II Receptor Blocker (ARB)",
        "typical_doses": ["20mg", "40mg", "80mg"],
        "max_single_dose_mg": 80,
        "max_daily_dose_mg": 80,
        "route": "Oral",
        "description": "Antihypertensive medication reducing cardiovascular risk."
    },
    {
        "id": "DRUG_016",
        "generic_name": "Losartan",
        "synonyms": ["Losar", "Losor", "losay", "Losar 25", "Losar 50"],
        "brands": ["Losar", "Losar 25", "Cozaar", "Repace", "Tozaar"],
        "class": "Angiotensin II Receptor Blocker (ARB)",
        "typical_doses": ["25mg", "50mg", "100mg"],
        "max_single_dose_mg": 100,
        "max_daily_dose_mg": 100,
        "route": "Oral",
        "description": "Treats hypertension and delays diabetic nephropathy progression."
    },
    {
        "id": "DRUG_017",
        "generic_name": "Aspirin",
        "synonyms": ["Acetylsalicylic Acid", "ASA"],
        "brands": ["Ecosprin", "Disprin", "Bayer", "Aspent"],
        "class": "Antiplatelet / Salicylate NSAID",
        "typical_doses": ["75mg", "81mg", "150mg", "325mg"],
        "max_single_dose_mg": 325,
        "max_daily_dose_mg": 325,  # For cardiovascular antiplatelet
        "route": "Oral",
        "description": "Antiplatelet agent preventing heart attacks and strokes."
    },
    {
        "id": "DRUG_018",
        "generic_name": "Clopidogrel",
        "synonyms": ["Clopid"],
        "brands": ["Plavix", "Clopilet", "Deplatt", "Ceruvit"],
        "class": "P2Y12 Platelet Inhibitor",
        "typical_doses": ["75mg", "300mg"],
        "max_single_dose_mg": 300,
        "max_daily_dose_mg": 75,
        "route": "Oral",
        "description": "Antiplatelet preventing atherothrombotic vascular events."
    },
    {
        "id": "DRUG_019",
        "generic_name": "Warfarin",
        "synonyms": ["Coumarin", "Warf"],
        "brands": ["Coumadin", "Uniwarfin", "Warf"],
        "class": "Vitamin K Antagonist Anticoagulant",
        "typical_doses": ["1mg", "2mg", "2.5mg", "5mg"],
        "max_single_dose_mg": 10,
        "max_daily_dose_mg": 10,
        "route": "Oral",
        "description": "Anticoagulant requiring close INR monitoring; high bleeding risk."
    },
    {
        "id": "DRUG_020",
        "generic_name": "Tramadol",
        "synonyms": ["Trama"],
        "brands": ["Ultram", "Tramazac", "Domadol", "Tramatas"],
        "class": "Opioid Analgesic",
        "typical_doses": ["50mg", "100mg"],
        "max_single_dose_mg": 100,
        "max_daily_dose_mg": 400,
        "route": "Oral / IV",
        "description": "Centrally acting synthetic opioid for moderate-to-severe pain."
    },
    {
        "id": "DRUG_021",
        "generic_name": "Metronidazole",
        "synonyms": ["Metro", "Metron"],
        "brands": ["Flagyl", "Metrogyl", "Metron"],
        "class": "Nitroimidazole Antimicrobial",
        "typical_doses": ["200mg", "400mg", "500mg"],
        "max_single_dose_mg": 500,
        "max_daily_dose_mg": 1500,
        "route": "Oral / IV",
        "description": "Treats anaerobic bacterial and parasitic protozoal infections."
    },
    {
        "id": "DRUG_022",
        "generic_name": "Doxycycline",
        "synonyms": ["Doxy"],
        "brands": ["Vibramycin", "Doxicip", "Microdox", "Doxt"],
        "class": "Tetracycline Antibiotic",
        "typical_doses": ["100mg"],
        "max_single_dose_mg": 100,
        "max_daily_dose_mg": 200,
        "route": "Oral",
        "description": "Tetracycline class antibiotic for respiratory, skin, and tick-borne infections."
    },
    {
        "id": "DRUG_023",
        "generic_name": "Salbutamol",
        "synonyms": ["Albuterol", "Asthalin"],
        "brands": ["Ventolin", "Asthalin", "ProAir", "Aerolin"],
        "class": "Short-acting Beta-2 Agonist (SABA)",
        "typical_doses": ["2mg", "4mg", "100mcg/puff"],
        "max_single_dose_mg": 4,
        "max_daily_dose_mg": 16,
        "route": "Inhalation / Oral",
        "description": "Bronchodilator for rapid relief of asthma and COPD bronchospasm."
    },
    {
        "id": "DRUG_024",
        "generic_name": "Ondansetron",
        "synonyms": ["Ondan", "Ondas"],
        "brands": ["Zofran", "Emeset", "Vomistop", "Periset"],
        "class": "5-HT3 Receptor Antagonist (Antiemetic)",
        "typical_doses": ["4mg", "8mg"],
        "max_single_dose_mg": 8,
        "max_daily_dose_mg": 24,
        "route": "Oral / IV",
        "description": "Prevents and treats nausea and vomiting."
    },
    {
        "id": "DRUG_025",
        "generic_name": "B-Complex Vitamins",
        "synonyms": ["Multivitamin", "B-Complex", "Vitamin B"],
        "brands": ["Becosules", "Neurobion", "Cobadex", "Optineuron"],
        "class": "Nutritional Supplement",
        "typical_doses": ["1 cap", "1 tab"],
        "max_single_dose_mg": 100,
        "max_daily_dose_mg": 100,
        "route": "Oral",
        "description": "Dietary supplement for nutritional deficiencies and convalescence."
    },
    {
        "id": "DRUG_026",
        "generic_name": "Calcium + Vitamin D3",
        "synonyms": ["Cal-D3", "Calcium Carb"],
        "brands": ["Shelcal", "Shelcal 500", "Caltrate", "Cipcal 500", "Ostocalcium"],
        "class": "Mineral & Vitamin Supplement",
        "typical_doses": ["500mg", "1000mg"],
        "max_single_dose_mg": 1000,
        "max_daily_dose_mg": 1500,
        "route": "Oral",
        "description": "Promotes bone density and prevents osteoporosis/osteopenia."
    },
    {
        "id": "DRUG_027",
        "generic_name": "Celecoxib",
        "synonyms": ["Celecox"],
        "brands": ["Celebrex", "Cobix", "Zycel"],
        "class": "COX-2 Selective NSAID",
        "typical_doses": ["100mg", "200mg"],
        "max_single_dose_mg": 200,
        "max_daily_dose_mg": 400,
        "route": "Oral",
        "description": "NSAID with lower gastric ulceration risk for arthritis."
    },
    {
        "id": "DRUG_028",
        "generic_name": "Citalopram",
        "synonyms": ["Citalo"],
        "brands": ["Celexa", "Cipramil", "Citadep"],
        "class": "SSRI Antidepressant",
        "typical_doses": ["10mg", "20mg", "40mg"],
        "max_single_dose_mg": 40,
        "max_daily_dose_mg": 40,
        "route": "Oral",
        "description": "Antidepressant for major depressive disorder."
    },
    {
        "id": "DRUG_029",
        "generic_name": "Clonidine",
        "synonyms": ["Catapres"],
        "brands": ["Catapres", "Arkamin", "Cloneon"],
        "class": "Alpha-2 Adrenergic Agonist",
        "typical_doses": ["0.1mg", "0.2mg", "100mcg"],
        "max_single_dose_mg": 0.3,
        "max_daily_dose_mg": 1.2,
        "route": "Oral",
        "description": "Centrally acting antihypertensive."
    },
    {
        "id": "DRUG_030",
        "generic_name": "Clonazepam",
        "synonyms": ["Clona", "Clonaz", "Lonazep", "Lonezep", "LoNezop", "Lonazep 1mg", "Lonezep 1mg"],
        "brands": ["Lonazep", "Lonezep", "Klonopin", "Rivotril", "ClonaFit", "Zapiz"],
        "class": "Benzodiazepine Anticonvulsant / Anxiolytic",
        "typical_doses": ["0.25mg", "0.5mg", "1mg", "2mg"],
        "max_single_dose_mg": 2,
        "max_daily_dose_mg": 4,
        "route": "Oral",
        "description": "Treats seizure disorders and panic attacks."
    },
    {
        "id": "DRUG_031",
        "generic_name": "Hydroxyzine",
        "synonyms": ["Hydroxyz"],
        "brands": ["Atarax", "Vistaril"],
        "class": "First-generation Antihistamine / Anxiolytic",
        "typical_doses": ["10mg", "25mg"],
        "max_single_dose_mg": 50,
        "max_daily_dose_mg": 100,
        "route": "Oral",
        "description": "Sedating antihistamine for pruritus, urticaria, and anxiety."
    },
    {
        "id": "DRUG_032",
        "generic_name": "Hydralazine",
        "synonyms": ["Hydral"],
        "brands": ["Apresoline", "Dinex"],
        "class": "Direct Vasodilator Antihypertensive",
        "typical_doses": ["25mg", "50mg"],
        "max_single_dose_mg": 50,
        "max_daily_dose_mg": 200,
        "route": "Oral / IV",
        "description": "Direct arterial vasodilator used in resistant hypertension and pre-eclampsia."
    },
    {
        "id": "DRUG_033",
        "generic_name": "Spironolactone",
        "synonyms": ["Spirono", "Spiron"],
        "brands": ["Aldactone", "Spiron", "Aldopur"],
        "class": "Potassium-Sparing Diuretic / Aldosterone Antagonist",
        "typical_doses": ["25mg", "50mg", "100mg"],
        "max_single_dose_mg": 100,
        "max_daily_dose_mg": 200,
        "route": "Oral",
        "description": "Treats heart failure, ascites, and hypertension while conserving potassium."
    },
    {
        "id": "DRUG_034",
        "generic_name": "Lisinopril",
        "synonyms": ["Lisino"],
        "brands": ["Prinivil", "Zestril", "Lipril"],
        "class": "ACE Inhibitor Antihypertensive",
        "typical_doses": ["5mg", "10mg", "20mg"],
        "max_single_dose_mg": 40,
        "max_daily_dose_mg": 40,
        "route": "Oral",
        "description": "ACE inhibitor treating hypertension and congestive heart failure."
    },
    {
        "id": "DRUG_035",
        "generic_name": "Digoxin",
        "synonyms": ["Digox"],
        "brands": ["Lanoxin", "Lanicor"],
        "class": "Cardiac Glycoside",
        "typical_doses": ["0.125mg", "0.25mg"],
        "max_single_dose_mg": 0.25,
        "max_daily_dose_mg": 0.25,
        "route": "Oral / IV",
        "description": "Inotropic agent for atrial fibrillation and severe heart failure; narrow therapeutic index."
    },
    {
        "id": "DRUG_036",
        "generic_name": "Itraconazole",
        "synonyms": ["Itra", "Itracon", "Itaspor", "Itaspr", "Itraspor", "Docowel", "Doclowel", "Qg igra", "Qag Higra", "Higra", "Itaspor 200", "Itaspor 200mg"],
        "brands": ["Itaspor", "Itaspor 200", "Canditral", "Sporanox", "Candiforce", "Itra", "Docowel", "Doclowel"],
        "class": "Triazole Systemic Antifungal",
        "typical_doses": ["200mg", "100mg"],
        "max_single_dose_mg": 200,
        "max_daily_dose_mg": 400,
        "route": "Oral",
        "description": "Systemic triazole antifungal for dermatophytosis, tinea/pityriasis versicolor, and onychomycosis."
    },
    {
        "id": "DRUG_037",
        "generic_name": "Eberconazole",
        "synonyms": ["Eber", "Ebernet", "Episcent", "Eberfun", "Ebrnet", "ung gly", "Ebernet Cream"],
        "brands": ["Ebernet", "Ebernet Cream", "Episcent", "Eberfun", "Ebcon", "Ebrnet"],
        "class": "Imidazole Topical Antifungal",
        "typical_doses": ["1% cream"],
        "max_single_dose_mg": 100,
        "max_daily_dose_mg": 200,
        "route": "Topical",
        "description": "Topical antifungal active against dermatophytes, yeasts, and Pityriasis versicolor."
    },
    {
        "id": "DRUG_038",
        "generic_name": "Terbinafine",
        "synonyms": ["Terbi", "Lamifin", "Lamisil", "Terbicip", "Langfa", "Lanofa", "Lamfa", "Lamfin", "Lanofa Caam", "Lamifin Cream"],
        "brands": ["Lamifin", "Lamifin Cream", "Lamisil", "Terbicip", "Sebifin", "Tyza", "Langfa", "Lanofa", "Lamfa", "Lamfin"],
        "class": "Allylamine Antifungal",
        "typical_doses": ["1% cream", "250mg"],
        "max_single_dose_mg": 250,
        "max_daily_dose_mg": 250,
        "route": "Topical / Oral",
        "description": "Inhibits squalene epoxidase; highly effective against dermatophytes and cutaneous mycoses."
    },
    {
        "id": "DRUG_039",
        "generic_name": "Ketoconazole",
        "synonyms": ["Keto", "Nizoclin", "Nizoral", "Ketocip", "Nizoc", "Nizocn"],
        "brands": ["Nizoclin", "Nizoclin Soap", "Nizoral", "Ketocip", "Danfree", "Nizoc", "Nizocn"],
        "class": "Antifungal Cleansing Agent",
        "typical_doses": ["2% soap", "2% shampoo", "200mg"],
        "max_single_dose_mg": 200,
        "max_daily_dose_mg": 400,
        "route": "Topical / Oral",
        "description": "Medicated soap/cleanser for Pityriasis versicolor, seborrheic dermatitis, and cutaneous candidiasis."
    },
    {
        "id": "DRUG_040",
        "generic_name": "Diclofenac + Paracetamol + Serratiopeptidase",
        "synonyms": ["Enzoflam", "Enzoflam Tablet", "Enzsflam", "Enzplam", "Enzo", "Diclofenac Paracetamol Serratiopeptidase", "Diclo-Para-Serratio"],
        "brands": ["Enzoflam", "Enzoflam D", "Alanz", "Seradic-P", "Cipzen-D", "Enzsflam", "Enzplam", "Enzo"],
        "class": "NSAID + Proteolytic Enzyme Analgesic / Anti-inflammatory",
        "typical_doses": ["Tablet", "50mg/325mg/15mg"],
        "max_single_dose_mg": 1000,
        "max_daily_dose_mg": 3000,
        "route": "Oral",
        "description": "Combination anti-inflammatory, analgesic, and proteolytic enzyme for dental pain, inflammation, and post-procedure swelling."
    },
    {
        "id": "DRUG_041",
        "generic_name": "Pantoprazole + Domperidone",
        "synonyms": ["Pan-D", "Pan D", "Pantocid-D", "Pantocid D", "Panto-D", "PanDLOnJ", "PanD", "Pantodac-D", "Pan-D 40", "Pan D 40", "Pan-40"],
        "brands": ["Pan-D", "Pantocid D", "Pantop-D", "Pan-D 40", "PanDLOnJ", "Pan D", "Pan-L", "Pan D 40mg", "Pan D Capsule", "Pan40"],
        "class": "Proton Pump Inhibitor (PPI) + Prokinetic Antacid",
        "typical_doses": ["40mg", "40mg/30mg"],
        "max_single_dose_mg": 40,
        "max_daily_dose_mg": 80,
        "route": "Oral",
        "description": "Gastroprotective combination of PPI (pantoprazole 40mg) and prokinetic (domperidone 30mg) for acid reflux and NSAID gastric protection."
    },
    {
        "id": "DRUG_042",
        "generic_name": "Chlorhexidine Gluconate",
        "synonyms": ["Hexigel", "Hexigel Gum Paint", "Chlorhexidine Gluconate Gel", "Chlorhexidine", "Herigel", "Herigel gum pant", "Hexigel gum paint", "Hexidine Gum Paint", "Herigel gum"],
        "brands": ["Hexigel", "Hexigel Gum Paint", "Hexidine", "Rexidin", "Eludril", "Herigel", "Herigel gum pant", "Hexigel paint", "Hexidine paint", "Hexigel gel"],
        "class": "Antiseptic / Anti-gingivitis Oral Gel",
        "typical_doses": ["1% w/w", "Gum Paint / Oral Gel"],
        "max_single_dose_mg": 0,
        "max_daily_dose_mg": 0,
        "route": "Topical (Gums / Oral Mucosa)",
        "description": "Antiseptic dental gel / gum paint for gingivitis, oral wounds, plaque control, and post-dental surgical care."
    },
    {
        "id": "DRUG_043",
        "generic_name": "Ketorolac Tromethamine",
        "synonyms": [
            "Ketorolac", "Ketorol", "Ketorel", "Toradol", "Ketanov", "Ketoflam",
            "Ketorol-DT", "Ketorol DT", "Ketorel-DT", "Ketorel DT", "Talo ketool",
            "Tolo ketorol", "Ketool", "Ketovol", "Ketoral"
        ],
        "brands": [
            "Ketorol-DT", "Ketorol", "Ketorel-DT", "Ketorel", "Toradol",
            "Ketanov", "Ketoflam", "Ketorol DT"
        ],
        "class": "NSAID (Potent Analgesic / Anti-inflammatory)",
        "typical_doses": ["10mg DT", "10mg", "Dispersible Tablet"],
        "max_single_dose_mg": 20,
        "max_daily_dose_mg": 40,
        "route": "Oral / Dispersible",
        "description": "Potent NSAID analgesic for moderate-to-severe acute pain (dental, surgical); typically prescribed as dispersible tablet for up to 5 days."
    },
    {
        "id": "DRUG_044",
        "generic_name": "Divalproex Sodium",
        "synonyms": [
            "Sodium Valproate", "Valproic Acid", "Valproate", "Divalproex", "Dicorate",
            "Dicorate ER", "Dicorate ER 250", "Divalrate", "Divalrate ER", "Depakote",
            "icrale", "icrale ER", "Dicrale ER", "Valparin", "Epilim"
        ],
        "brands": [
            "Dicorate ER", "Dicorate ER 250", "Dicorate", "Divalrate ER", "Depakote",
            "Valparin", "Epilim", "Encorate", "Encorate Chrono"
        ],
        "class": "Anticonvulsant / Mood Stabilizer / Migraine Prophylaxis",
        "typical_doses": ["250mg", "500mg", "125mg", "750mg", "1000mg"],
        "max_single_dose_mg": 1000,
        "max_daily_dose_mg": 2500,
        "route": "Oral",
        "description": "First-line broad-spectrum anticonvulsant, mood stabilizer for bipolar disorder, and migraine headache prophylaxis."
    },
    {
        "id": "DRUG_045",
        "generic_name": "Metformin + Glibenclamide",
        "synonyms": [
            "Glucored Forte", "Glucored", "Glucovance", "Glibenclamide Metformin",
            "Gucrea", "Gucrea Fme", "Cucrea", "Cucrea Come", "Glucored-Forte"
        ],
        "brands": [
            "Glucored Forte", "Glucored", "Glucovance", "Glibomet", "Daonil-M"
        ],
        "class": "Biguanide + Sulfonylurea Antidiabetic",
        "typical_doses": ["Forte (500mg/5mg)", "500mg/2.5mg", "500mg/5mg"],
        "max_single_dose_mg": 1000,
        "max_daily_dose_mg": 2000,
        "route": "Oral",
        "description": "Dual oral hypoglycemic agent combining insulin sensitizer (metformin) and insulin secretagogue (glibenclamide) for Type 2 diabetes."
    },
    {
        "id": "DRUG_046",
        "generic_name": "Escitalopram + Clonazepam",
        "synonyms": [
            "Localm PLUS", "Localm Plus", "Lacalm PLUS", "Lacalm Plus", "Lecalm PLUS",
            "Nexito Plus", "Clonafit Plus", "Lacam", "Lacam PLUS", "Lacalm Pws"
        ],
        "brands": [
            "Localm PLUS", "Nexito Plus", "Clonafit Plus", "Es-Citadep Plus", "Szetalo Plus", "Lacalm Plus"
        ],
        "class": "SSRI + Benzodiazepine Anxiolytic",
        "typical_doses": ["10mg/0.5mg", "5mg/0.5mg", "Standard Dose"],
        "max_single_dose_mg": 20,
        "max_daily_dose_mg": 20,
        "route": "Oral",
        "description": "Fixed-dose combination of SSRI antidepressant and benzodiazepine for generalized anxiety disorder, panic attacks, and comorbid depression."
    },
    {
        "id": "DRUG_047",
        "generic_name": "Losartan + Hydrochlorothiazide",
        "synonyms": [
            "Losar H", "Losar-H", "Repace H", "Repace-H", "Cosart-H", "Hyzaar",
            "Losor H", "LosorH", "Hoodoy", "Repace H component"
        ],
        "brands": [
            "Losar H", "Repace H", "Cosart H", "Hyzaar", "Tozaar H"
        ],
        "class": "Angiotensin II Receptor Blocker (ARB) + Thiazide Diuretic",
        "typical_doses": ["50mg/12.5mg", "50mg", "100mg/25mg"],
        "max_single_dose_mg": 100,
        "max_daily_dose_mg": 100,
        "route": "Oral",
        "description": "Combination antihypertensive providing synergistic blood pressure reduction via ARB renin-angiotensin blockade and thiazide volume reduction."
    },
    {
        "id": "DRUG_048",
        "generic_name": "Paracetamol (Paediatric Suspension)",
        "synonyms": [
            "Calpol", "Calpol 250", "Calpol 120", "Calpol Drops", "Calpol Syrup", "P-250", "Crocin DS",
            "Capol", "Colpol", "Calpol@250", "Calpol 250/5", "P-120", "Pacimol", "Pyrigesic", "59PCAPOL"
        ],
        "brands": [
            "Calpol", "Calpol 250", "Calpol 120", "Crocin DS", "P-250", "Dolo 120", "Dolo 250"
        ],
        "class": "Antipyretic / Analgesic (Paediatric)",
        "typical_doses": ["250mg/5ml", "120mg/5ml", "100mg/ml", "4ml", "5ml"],
        "max_single_dose_mg": 500,
        "max_daily_dose_mg": 2000,
        "route": "Oral",
        "description": "Paediatric antipyretic and analgesic suspension for fever, viral illnesses, and mild-to-moderate pain."
    },
    {
        "id": "DRUG_049",
        "generic_name": "Phenylephrine + Chlorpheniramine Maleate",
        "synonyms": [
            "Delcon", "Delcon Syrup", "Delcon Drops", "Maxtra", "Solvin Cold", "Febrex Plus",
            "Elcon", "Delcon 3ml", "Delcon syp", "Sinarest Paediatric", "DELCON"
        ],
        "brands": [
            "Delcon", "Delcon Syrup", "Maxtra", "Solvin Cold", "Febrex Plus", "Sinarest"
        ],
        "class": "Decongestant / Antihistamine Paediatric Syrup",
        "typical_doses": ["3ml", "2.5ml", "5ml"],
        "max_single_dose_mg": 10,
        "max_daily_dose_mg": 30,
        "route": "Oral",
        "description": "Paediatric nasal decongestant and antihistamine syrup for upper respiratory tract infection (URTI), rhinorrhea, and allergic rhinitis."
    },
    {
        "id": "DRUG_050",
        "generic_name": "Levosalbutamol",
        "synonyms": [
            "Levolin", "Levolin Syrup", "Levolin 1mg/5ml", "Levolin 3ml", "Levolin Inhaler",
            "Levohn", "Levoln", "Asthalin", "Levolin syp", "LEVOHN"
        ],
        "brands": [
            "Levolin", "Levolin Syrup", "Levolin 1mg/5ml", "Asthalin", "Ventorlin"
        ],
        "class": "Bronchodilator (Selective Beta-2 Agonist)",
        "typical_doses": ["1mg/5ml", "3ml", "2.5ml", "5ml"],
        "max_single_dose_mg": 2,
        "max_daily_dose_mg": 6,
        "route": "Oral",
        "description": "Paediatric selective beta-2 adrenoreceptor agonist for bronchospasm, asthma, bronchitis, and reactive airway disease."
    },
    {
        "id": "DRUG_051",
        "generic_name": "Mefenamic Acid",
        "synonyms": [
            "Meftal-P", "Meftal P", "Meftal-P Suspension", "Meftal", "Meftal 100/5",
            "Mefrl-P", "Meftbl-P", "Meftl-P", "Meftal-P 100", "Ponstan", "mEFRL-P"
        ],
        "brands": [
            "Meftal-P", "Meftal-P Suspension", "Meftal", "Mefkind-P", "Ponstan"
        ],
        "class": "NSAID / Paediatric Antipyretic & Analgesic",
        "typical_doses": ["100mg/5ml", "3ml", "2.5ml", "5ml"],
        "max_single_dose_mg": 250,
        "max_daily_dose_mg": 500,
        "route": "Oral",
        "description": "Paediatric non-steroidal anti-inflammatory suspension for high fever refractory to paracetamol and acute pain relief."
    },
    {
        "id": "DRUG_052",
        "generic_name": "Ibuprofen + Paracetamol",
        "synonyms": ["Combiflam", "Combiflam Tablet", "Ibugesic Plus", "Flexon", "Ibu-Para", "Combiflan"],
        "brands": ["Combiflam", "Ibugesic Plus", "Flexon", "Brufen Plus"],
        "class": "NSAID + Analgesic Combination",
        "typical_doses": ["400mg/325mg", "1 tablet"],
        "max_single_dose_mg": 800,
        "max_daily_dose_mg": 2400,
        "route": "Oral",
        "description": "Combination anti-inflammatory and antipyretic for dental, musculoskeletal pain, and fever."
    },
    {
        "id": "DRUG_053",
        "generic_name": "Fexofenadine",
        "synonyms": ["Allegra", "Allegra 120", "Allegra 180", "Allegra 120mg", "Allegra 180mg", "Fexova", "Fexy", "Telfast"],
        "brands": ["Allegra", "Allegra 120", "Allegra 180", "Fexy", "Fexova", "Histafree"],
        "class": "Second-Generation Antihistamine",
        "typical_doses": ["120mg", "180mg", "60mg"],
        "max_single_dose_mg": 180,
        "max_daily_dose_mg": 180,
        "route": "Oral",
        "description": "Non-sedating antihistamine for allergic rhinitis, sneezing, urticaria, and allergies."
    },
    {
        "id": "DRUG_054",
        "generic_name": "Levosalbutamol + Ambroxol + Guaiphenesin",
        "synonyms": ["Ascoril", "Ascoril LS", "Ascoril D", "Ascoril Syrup", "Bro-Zedex", "Macbery XT", "Ascoril-D", "Ascoril-LS"],
        "brands": ["Ascoril LS", "Ascoril D", "Ascoril", "Bro-Zedex", "Macbery"],
        "class": "Bronchodilator & Mucolytic Expectorant",
        "typical_doses": ["5ml", "10ml", "Syrup"],
        "max_single_dose_mg": 10,
        "max_daily_dose_mg": 30,
        "route": "Oral",
        "description": "Triple-action cough syrup for productive cough, bronchitis, and bronchospasm."
    },
    {
        "id": "DRUG_055",
        "generic_name": "Multivitamins + Minerals + Trace Elements",
        "synonyms": ["Supradyn", "Supradyn Daily", "Zincovit", "Becosules", "A to Z", "Revital", "Supradin"],
        "brands": ["Supradyn", "Zincovit", "Becosules", "Revital H", "A to Z NS"],
        "class": "Nutritional Supplement / Multivitamin",
        "typical_doses": ["1 tablet", "1 capsule", "OD"],
        "max_single_dose_mg": 0,
        "max_daily_dose_mg": 0,
        "route": "Oral",
        "description": "Comprehensive multivitamin and mineral formulation for immunity, convalescence, and daily nutritional support."
    },
    {
        "id": "DRUG_056",
        "generic_name": "Folic Acid",
        "synonyms": ["Folvite", "Folvite 5mg", "Folv", "Folv 5", "Folic Acid 5mg", "Folinal", "Folfit"],
        "brands": ["Folvite", "Folvite 5mg", "Folinal", "Folfit", "Folichem"],
        "class": "Vitamin Supplement / Hematopoietic Agent",
        "typical_doses": ["5mg", "1mg"],
        "max_single_dose_mg": 5,
        "max_daily_dose_mg": 5,
        "route": "Oral",
        "description": "Water-soluble B-vitamin (B9) essential for DNA synthesis, red blood cell formation, and neural tube defect prevention."
    },
    {
        "id": "DRUG_057",
        "generic_name": "Montelukast + Levocetirizine",
        "synonyms": ["Montek LC", "Montair LC", "Montek-LC", "Montair-LC", "Telekast-L", "Levocet-M", "Romilast-L", "Monticope"],
        "brands": ["Montek LC", "Montair LC", "Telekast-L", "Levocet M", "Romilast L"],
        "class": "Leukotriene Receptor Antagonist + Antihistamine",
        "typical_doses": ["10mg/5mg", "1 tablet"],
        "max_single_dose_mg": 15,
        "max_daily_dose_mg": 15,
        "route": "Oral",
        "description": "Dual anti-allergic combination for allergic rhinitis, nocturnal asthma symptoms, and persistent bronchospasm."
    },
    {
        "id": "DRUG_058",
        "generic_name": "Cefixime",
        "synonyms": ["Taxim-O", "Zifi", "Cefolac", "Mahacef", "Taxim O 200", "Zifi 200", "Cefixime 200", "Cefix"],
        "brands": ["Taxim-O", "Zifi", "Cefolac", "Mahacef", "Omnicef"],
        "class": "Third-Generation Cephalosporin Antibiotic",
        "typical_doses": ["200mg", "100mg", "400mg"],
        "max_single_dose_mg": 400,
        "max_daily_dose_mg": 400,
        "route": "Oral",
        "description": "Third-generation oral cephalosporin antibiotic for respiratory tract infections, typhoid, and urinary tract infections."
    },
    {
        "id": "DRUG_059",
        "generic_name": "Azithromycin",
        "synonyms": ["Azithral", "Azee", "Azicip", "Zithrox", "Azithral 500", "Azee 500", "Azithromycin 500", "Zithromax"],
        "brands": ["Azithral", "Azee", "Azicip", "Zithrox", "Zady"],
        "class": "Macrolide Antibiotic",
        "typical_doses": ["500mg", "250mg"],
        "max_single_dose_mg": 500,
        "max_daily_dose_mg": 500,
        "route": "Oral",
        "description": "Broad-spectrum azalide/macrolide antibiotic for respiratory tract, ear, throat, and skin infections; typically a 3- to 5-day course."
    },
    {
        "id": "DRUG_060",
        "generic_name": "Telmisartan",
        "synonyms": ["Telma", "Telma 40", "Telma 20", "Telma 80", "Telmikind", "Telsartan", "Micardis", "Telvas"],
        "brands": ["Telma", "Telma 40", "Telmikind", "Telsartan", "Telvas", "Micardis"],
        "class": "Angiotensin II Receptor Blocker (ARB)",
        "typical_doses": ["40mg", "20mg", "80mg"],
        "max_single_dose_mg": 80,
        "max_daily_dose_mg": 80,
        "route": "Oral",
        "description": "Long-acting ARB for essential hypertension and cardiovascular risk reduction."
    },
    {
        "id": "DRUG_061",
        "generic_name": "Aspirin",
        "synonyms": ["Ecosprin", "Ecosprin 75", "Ecosprin 150", "Disprin", "Loprin", "Acetylsalicylic Acid", "ASA"],
        "brands": ["Ecosprin", "Ecosprin 75", "Ecosprin 150", "Disprin", "Loprin"],
        "class": "Antiplatelet Agent / Salicylate",
        "typical_doses": ["75mg", "150mg", "325mg"],
        "max_single_dose_mg": 325,
        "max_daily_dose_mg": 325,
        "route": "Oral",
        "description": "Low-dose antiplatelet therapy for secondary prevention of myocardial infarction, stroke, and cardiovascular disease."
    },
    {
        "id": "DRUG_062",
        "generic_name": "Levothyroxine Sodium",
        "synonyms": ["Thyronorm", "Eltroxin", "Thyrox", "Synthroid", "Thyronorm 50mcg", "Thyronorm 25mcg", "Thyronorm 100mcg"],
        "brands": ["Thyronorm", "Eltroxin", "Thyrox", "Synthroid"],
        "class": "Synthetic Thyroid Hormone",
        "typical_doses": ["25mcg", "50mcg", "75mcg", "100mcg", "12.5mcg"],
        "max_single_dose_mg": 0.2,
        "max_daily_dose_mg": 0.2,
        "route": "Oral (Empty Stomach)",
        "description": "Thyroid hormone replacement therapy for hypothyroidism and goitre."
    },
    {
        "id": "DRUG_063",
        "generic_name": "Calcium Carbonate + Vitamin D3",
        "synonyms": ["Shelcal", "Shelcal 500", "Cipcal", "Gemcal", "Shelcal-500", "Cipcal 500", "Cal-D"],
        "brands": ["Shelcal", "Shelcal 500", "Cipcal", "Gemcal", "Supracal"],
        "class": "Mineral & Vitamin Supplement",
        "typical_doses": ["500mg", "500mg/250IU", "1 tablet"],
        "max_single_dose_mg": 1000,
        "max_daily_dose_mg": 1500,
        "route": "Oral",
        "description": "Essential calcium and cholecalciferol formulation for bone health, osteoporosis, and calcium deficiency."
    },
    {
        "id": "DRUG_064",
        "generic_name": "Paracetamol + Phenylephrine + Chlorpheniramine",
        "synonyms": ["Sinarest", "Cheston Cold", "Wikoryl", "Solvin Cold", "Sinarest Tablet", "Cheston Cold Tablet"],
        "brands": ["Sinarest", "Cheston Cold", "Wikoryl", "Solvin Cold", "Febrex Plus"],
        "class": "Antipyretic / Decongestant Combination",
        "typical_doses": ["1 tablet", "500mg/10mg/2mg"],
        "max_single_dose_mg": 1000,
        "max_daily_dose_mg": 3000,
        "route": "Oral",
        "description": "Comprehensive cold and flu formulation relieving fever, headache, nasal congestion, and allergic symptoms."
    },
    {
        "id": "DRUG_065",
        "generic_name": "Mefenamic Acid + Dicyclomine",
        "synonyms": ["Meftal Spas", "Meftal-Spas", "Cyclopam", "Spasmo-Proxyvon", "Mefkind-Spas", "Spasmonil"],
        "brands": ["Meftal Spas", "Meftal-Spas", "Cyclopam", "Spasmo-Proxyvon"],
        "class": "Antispasmodic & NSAID Combination",
        "typical_doses": ["250mg/10mg", "500mg/20mg", "1 tablet"],
        "max_single_dose_mg": 500,
        "max_daily_dose_mg": 1500,
        "route": "Oral",
        "description": "Dual antispasmodic and analgesic for smooth muscle spasm, primary dysmenorrhea, and abdominal colic."
    },
    {
        "id": "DRUG_066",
        "generic_name": "Ascorbic Acid (Vitamin C)",
        "synonyms": ["Limcee", "Limcee 500", "Celin", "Celin 500", "Suckcee", "Vitamin C"],
        "brands": ["Limcee", "Celin", "Suckcee", "Redoxon"],
        "class": "Water-Soluble Vitamin / Antioxidant",
        "typical_doses": ["500mg", "1000mg"],
        "max_single_dose_mg": 1000,
        "max_daily_dose_mg": 2000,
        "route": "Oral",
        "description": "Immune support, wound healing, collagen synthesis, and scurvy prevention."
    },
    {
        "id": "DRUG_067",
        "generic_name": "Theophylline + Etofylline",
        "synonyms": ["Deriphyllin", "Deriphyllin Retard", "Deriphyllin 150", "Deriphyllin 300", "Deriphyllin-Retard", "Deriphyllin OD"],
        "brands": ["Deriphyllin", "Deriphyllin Retard", "Deriphyllin 150", "Deriphyllin 300"],
        "class": "Xanthine Bronchodilator",
        "typical_doses": ["150mg", "300mg"],
        "max_single_dose_mg": 300,
        "max_daily_dose_mg": 600,
        "route": "Oral / IV",
        "description": "Bronchodilator relaxing airway smooth muscle for asthma, chronic obstructive pulmonary disease (COPD), and chronic bronchitis."
    },
    {
        "id": "DRUG_068",
        "generic_name": "Levosalbutamol + Ipratropium Bromide",
        "synonyms": ["Duolin", "Duolin Respules", "Duolin Inhaler", "Duolin-Respules", "Duolin Rotacaps"],
        "brands": ["Duolin", "Duolin Respules", "Duolin Inhaler", "Combivent"],
        "class": "Dual Bronchodilator (SABA + SAMA)",
        "typical_doses": ["1.25mg/500mcg", "Respule", "Inhalation"],
        "max_single_dose_mg": 2.5,
        "max_daily_dose_mg": 5.0,
        "route": "Inhalation / Nebulization",
        "description": "Dual complementary bronchodilation for acute exacerbations of asthma and severe COPD."
    },
    {
        "id": "DRUG_069",
        "generic_name": "Budesonide",
        "synonyms": ["Budecort", "Budecort Respules", "Budecort Inhaler", "Budecort 0.5", "Budecort 1mg", "Pulmicort"],
        "brands": ["Budecort", "Budecort Respules", "Budecort Inhaler", "Pulmicort"],
        "class": "Inhaled Corticosteroid",
        "typical_doses": ["0.5mg", "1mg", "200mcg", "400mcg"],
        "max_single_dose_mg": 1.0,
        "max_daily_dose_mg": 2.0,
        "route": "Inhalation / Nebulization",
        "description": "Inhaled corticosteroid with potent local anti-inflammatory action for maintenance control of persistent asthma and COPD."
    },
    {
        "id": "DRUG_070",
        "generic_name": "Ranitidine HCl",
        "synonyms": ["Rantac", "Rantac 150", "Aciloc", "Aciloc 150", "Zinetac", "Histac", "Zantac"],
        "brands": ["Rantac", "Aciloc", "Zinetac", "Histac", "Rantac 150", "Aciloc 150"],
        "class": "H2 Receptor Antagonist",
        "typical_doses": ["150mg", "300mg"],
        "max_single_dose_mg": 150,
        "max_daily_dose_mg": 300,
        "route": "Oral / IV",
        "description": "H2 blocker reducing gastric acid production for gastritis, peptic ulcers, and acid indigestion."
    },
    {
        "id": "DRUG_071",
        "generic_name": "Aceclofenac + Paracetamol + Serratiopeptidase",
        "synonyms": ["Zerodol-SP", "Zerodol SP", "Zerodol-P", "Hifenac-SP", "Aceclo-SP", "Signoflam", "Zerodol"],
        "brands": ["Zerodol-SP", "Zerodol SP", "Hifenac SP", "Signoflam", "Zerodol P"],
        "class": "NSAID + Analgesic + Proteolytic Enzyme",
        "typical_doses": ["100mg/325mg/15mg", "1 tablet"],
        "max_single_dose_mg": 200,
        "max_daily_dose_mg": 400,
        "route": "Oral",
        "description": "Potent triple combination for post-operative pain, dental inflammation, arthritis, and tissue trauma."
    },
    {
        "id": "DRUG_072",
        "generic_name": "Trypsin + Chymotrypsin",
        "synonyms": ["Chymoral Forte", "Chymoral", "Chymoral Plus", "Chymoral-Forte", "Enzomac"],
        "brands": ["Chymoral Forte", "Chymoral", "Chymoral Plus"],
        "class": "Proteolytic Enzyme Anti-inflammatory",
        "typical_doses": ["100,000 Armour Units", "1 tablet"],
        "max_single_dose_mg": 200000,
        "max_daily_dose_mg": 400000,
        "route": "Oral (Before Meals)",
        "description": "Oral enzyme formulation accelerating resolution of traumatic and post-surgical edema and hematoma."
    },
    {
        "id": "DRUG_073",
        "generic_name": "Ciprofloxacin",
        "synonyms": ["Cifran", "Ciplox", "Cifran 500", "Ciplox 500", "Cipro", "Ciprobid"],
        "brands": ["Cifran", "Ciplox", "Ciprobid", "Cifran 500", "Ciplox 500"],
        "class": "Fluoroquinolone Antibiotic",
        "typical_doses": ["500mg", "250mg", "750mg"],
        "max_single_dose_mg": 750,
        "max_daily_dose_mg": 1500,
        "route": "Oral / IV / Eye-Ear Drops",
        "description": "Broad-spectrum fluoroquinolone antibiotic for bacterial urinary tract, gastrointestinal, bone, and skin infections."
    },
    {
        "id": "DRUG_074",
        "generic_name": "Ofloxacin",
        "synonyms": ["Oflox", "Zenflox", "Oflox 200", "Zenflox 200", "Tarivid", "Ofloxacin 200"],
        "brands": ["Oflox", "Zenflox", "Tarivid", "Oflox 200"],
        "class": "Fluoroquinolone Antibiotic",
        "typical_doses": ["200mg", "400mg"],
        "max_single_dose_mg": 400,
        "max_daily_dose_mg": 800,
        "route": "Oral / IV",
        "description": "Broad-spectrum fluoroquinolone for pneumonia, bronchitis, skin, and pelvic infections."
    },
    {
        "id": "DRUG_075",
        "generic_name": "Norfloxacin + Tinidazole",
        "synonyms": ["Norflox-TZ", "Norflox TZ", "Norflox 400", "Normaxin-TZ", "Norflox-T"],
        "brands": ["Norflox-TZ", "Norflox TZ", "Norflox", "Normaxin TZ"],
        "class": "Fluoroquinolone + Antiprotozoal Antibiotic",
        "typical_doses": ["400mg/600mg", "1 tablet"],
        "max_single_dose_mg": 800,
        "max_daily_dose_mg": 1600,
        "route": "Oral",
        "description": "Synergistic antibacterial and antiprotozoal for infectious gastroenteritis, amoebiasis, and diarrhea."
    },
    {
        "id": "DRUG_076",
        "generic_name": "Cephalexin",
        "synonyms": ["Sporidex", "Phexin", "Sporidex 500", "Phexin 500", "Keflex", "Cephadex"],
        "brands": ["Sporidex", "Phexin", "Keflex", "Sporidex 500"],
        "class": "First-Generation Cephalosporin Antibiotic",
        "typical_doses": ["500mg", "250mg"],
        "max_single_dose_mg": 1000,
        "max_daily_dose_mg": 4000,
        "route": "Oral",
        "description": "First-generation cephalosporin for skin and soft tissue, streptococcal pharyngitis, and urinary tract infections."
    },
    {
        "id": "DRUG_077",
        "generic_name": "Rosuvastatin",
        "synonyms": ["Rosuvas", "Rosuvas 10", "Rozucor", "Crestor", "Rosave", "Rosuvas 20", "Rosuvastatin 10"],
        "brands": ["Rosuvas", "Rozucor", "Crestor", "Rosave"],
        "class": "HMG-CoA Reductase Inhibitor (Statin)",
        "typical_doses": ["10mg", "20mg", "5mg", "40mg"],
        "max_single_dose_mg": 40,
        "max_daily_dose_mg": 40,
        "route": "Oral",
        "description": "High-intensity statin reducing LDL cholesterol and cardiovascular morbidity."
    },
    {
        "id": "DRUG_078",
        "generic_name": "Sitagliptin",
        "synonyms": ["Januvia", "Januvia 100", "Zita", "Zita 100", "Istavel", "Sitagliptin 100"],
        "brands": ["Januvia", "Zita", "Istavel"],
        "class": "DPP-4 Inhibitor Antidiabetic",
        "typical_doses": ["100mg", "50mg"],
        "max_single_dose_mg": 100,
        "max_daily_dose_mg": 100,
        "route": "Oral",
        "description": "Incretin enhancer (DPP-4 inhibitor) providing glucose-dependent insulin secretion for Type 2 diabetes."
    },
    {
        "id": "DRUG_079",
        "generic_name": "Dapagliflozin",
        "synonyms": ["Forxiga", "Forxiga 10", "Oxra", "Brenzys", "Dapagliflozin 10"],
        "brands": ["Forxiga", "Oxra", "Farxiga"],
        "class": "SGLT2 Inhibitor Antidiabetic",
        "typical_doses": ["10mg", "5mg"],
        "max_single_dose_mg": 10,
        "max_daily_dose_mg": 10,
        "route": "Oral",
        "description": "SGLT2 inhibitor promoting urinary glucose excretion with cardio-renal protection."
    },
    {
        "id": "DRUG_080",
        "generic_name": "Aluminium Hydroxide + Magnesium Hydroxide + Simethicone",
        "synonyms": ["Gelusil", "Digene", "Mucaine", "Gelusil MPS", "Digene Gel", "Antacid Gel"],
        "brands": ["Gelusil", "Digene", "Mucaine Gel", "Gelusil MPS"],
        "class": "Antacid & Antiflatulent",
        "typical_doses": ["10ml", "1-2 tablets"],
        "max_single_dose_mg": 0,
        "max_daily_dose_mg": 0,
        "route": "Oral",
        "description": "Neutralizes stomach acid and relieves bloating, heartburn, and hyperacidity."
    },
    {
        "id": "DRUG_081",
        "generic_name": "Povidone Iodine",
        "synonyms": ["Betadine", "Betadine Gargle", "Betadine Ointment", "Wokadine", "Betadine 2%", "Betadine Solution"],
        "brands": ["Betadine", "Wokadine", "Betasept"],
        "class": "Broad-Spectrum Antiseptic & Microbicide",
        "typical_doses": ["2% gargle", "5% ointment", "10% solution"],
        "max_single_dose_mg": 0,
        "max_daily_dose_mg": 0,
        "route": "Topical / Oral Gargle",
        "description": "Broad-spectrum antiseptic for pharyngitis gargling, superficial wound dressing, and surgical scrub."
    },
    {
        "id": "DRUG_082",
        "generic_name": "Triamcinolone Acetonide",
        "synonyms": ["Kenacort", "Kenacort 0.1%", "Tess Buccal Paste", "Orabase", "Kenalog"],
        "brands": ["Kenacort", "Tess Buccal Paste", "Kenalog"],
        "class": "Topical Corticosteroid",
        "typical_doses": ["0.1% oral paste"],
        "max_single_dose_mg": 0,
        "max_daily_dose_mg": 0,
        "route": "Topical (Oral Mucosa)",
        "description": "Anti-inflammatory adhesive paste for aphthous stomatitis and oral ulcerative lesions."
    },
    {
        "id": "DRUG_083",
        "generic_name": "Clotrimazole",
        "synonyms": ["Candid", "Candid Mouth Paint", "Candid-B", "Canesten", "Candid 1%"],
        "brands": ["Candid", "Candid Mouth Paint", "Canesten", "Candid-B"],
        "class": "Topical Imidazole Antifungal",
        "typical_doses": ["1% mouth paint", "1% cream"],
        "max_single_dose_mg": 0,
        "max_daily_dose_mg": 0,
        "route": "Topical (Oral Mucosa / Skin)",
        "description": "Topical antifungal for oral candidiasis (thrush) and cutaneous fungal infections."
    },
    {
        "id": "DRUG_084",
        "generic_name": "Xylometazoline HCl",
        "synonyms": ["Otrivin", "Otrivin Nasal Drops", "Nasivion", "Xylomist", "Otrivin 0.1%"],
        "brands": ["Otrivin", "Nasivion", "Xylomist"],
        "class": "Nasal Decongestant (Alpha-Adrenergic Agonist)",
        "typical_doses": ["0.1% drops/spray", "0.05% paediatric"],
        "max_single_dose_mg": 0,
        "max_daily_dose_mg": 0,
        "route": "Nasal",
        "description": "Topical vasoconstrictor relieving severe nasal congestion and sinus blockage."
    },
    {
        "id": "DRUG_085",
        "generic_name": "Levocetirizine",
        "synonyms": ["Levocet", "Levocet 5", "1-AL", "Vozet", "Levocetirizine 5mg", "Xyzal"],
        "brands": ["Levocet", "1-AL", "Vozet", "Xyzal", "Levocet 5"],
        "class": "Second-Generation Antihistamine",
        "typical_doses": ["5mg", "2.5mg"],
        "max_single_dose_mg": 5,
        "max_daily_dose_mg": 5,
        "route": "Oral",
        "description": "Potent non-sedating H1 receptor antagonist for allergic rhinitis, ocular allergy, and chronic urticaria."
    },
    {
        "id": "DRUG_086",
        "generic_name": "Cefpodoxime Proxetil",
        "synonyms": ["Gudcef", "Gudcef 200", "Monocef-O", "Monocef O", "Macpod", "Doxcef", "Cefoprox", "Gudcef 100", "Macpod 200"],
        "brands": ["Gudcef", "Monocef-O", "Macpod", "Doxcef", "Cefoprox"],
        "class": "Third-Generation Cephalosporin Antibiotic",
        "typical_doses": ["200mg", "100mg"],
        "max_single_dose_mg": 400,
        "max_daily_dose_mg": 400,
        "route": "Oral",
        "description": "Oral third-generation cephalosporin for acute otitis media, pharyngitis, tonsillitis, pneumonia, and UTIs."
    }
]

# Soundex implementation for phonetic fuzzy matching of mangled cursive OCR
def soundex(token: str) -> str:
    """Computes American Soundex code for a word token."""
    token = re.sub(r'[^A-Za-z]', '', token).upper()
    if not token:
        return "0000"
    
    first = token[0]
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    
    encoded = [first]
    prev = mapping.get(first, '0')
    
    for char in token[1:]:
        code = mapping.get(char, '0')
        if code != '0' and code != prev:
            encoded.append(code)
        prev = code
        if len(encoded) == 4:
            break
            
    while len(encoded) < 4:
        encoded.append('0')
        
    return "".join(encoded[:4])


class DrugDatabase:
    """Formulary lookup, validation, and fuzzy search engine."""

    def __init__(self, drugs: Optional[List[Dict[str, Any]]] = None):
        self.drugs = drugs or DRUG_DATABASE
        # Build search index
        self.search_index: List[Tuple[str, str, Dict[str, Any]]] = []
        # (normalized_term, soundex_code, drug_dict)
        for drug in self.drugs:
            # Generic
            gen_norm = self._normalize(drug["generic_name"])
            self.search_index.append((gen_norm, soundex(gen_norm), drug))
            # Synonyms
            for syn in drug.get("synonyms", []):
                syn_norm = self._normalize(syn)
                self.search_index.append((syn_norm, soundex(syn_norm), drug))
            # Brands
            for b in drug.get("brands", []):
                b_norm = self._normalize(b)
                self.search_index.append((b_norm, soundex(b_norm), drug))

    def _normalize(self, text: str) -> str:
        """Removes special characters and extra spaces for matching."""
        return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower().strip()

    STOPWORDS = {
        "for", "the", "and", "with", "take", "apply", "daily", "days", "day", "morn", 
        "morning", "night", "bath", "bedtime", "oral", "topical", "water", "food", 
        "after", "before", "meal", "meals", "dose", "doses", "each", "every", "time", 
        "times", "hour", "hours", "week", "weeks", "month", "months", "soap", "cream", 
        "tab", "tabs", "tablet", "tablets", "cap", "caps", "capsule", "capsules", 
        "syp", "syrup", "inj", "injection", "drops", "drop", "oint", "ointment", 
        "lotion", "gel", "wash", "powder", "spray", "solution", "susp", "suspension", 
        "use", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", 
        "ten", "sos", "prn", "stat", "tid", "bid", "qid", "od", "bd", "tds", "hs", 
        "qhs", "ac", "pc", "po", "la", "topical bath", "opc", "to", "in", "on", 
        "at", "by", "from", "of", "as", "is", "sunday", "closed", "sundayclosed", "monday",
        "massage", "massoqe", "paint", "pant", "gum", "gums", "grum", "adv", "adu",
        "advise", "advice", "rest", "rtpe", "rtfe", "rtpe4l", "rtfe4l", "dpox", "x5elay",
        "xlweek", "xiweek", "ashutosh", "mishra", "dental", "clinic", "bds", "surgeon",
        "elay", "dyst", "dys", "elav", "sec", "bridge", "nagar", "vasanth", "opp",
        "an", "x30an", "x3oay", "oay", "oan", "tolo", "talo", "prasao", "shyam", "prasad",
        "smile", "care", "implant", "chandanagar", "hyderabad", "super", "specialty",
        "center", "centre", "telangana", "consultant", "kumar", "sirisha", "praveen",
        "weight", "wt", "kg", "kgs", "height", "ht", "cm", "age", "gender", "cender", "sex",
        "clinical", "description", "urti", "rr", "rs", "bp", "pulse", "temp", "vitals",
        "name", "patient", "date", "findings", "history", "examination"
    }

    def find_match(self, query: str, threshold: float = 75.0) -> Optional[Dict[str, Any]]:
        """Matches a messy OCR extracted drug name against the drug database.
        
        Uses a composite score combining:
        1. Token Sort Ratio (Levenshtein)
        2. Partial Ratio (handles prefixes/short forms like 'Amox' -> 'Amoxicillin')
        3. Phonetic Soundex bonus for cursive OCR distortions (e.g. 'Paracefamol' vs 'Paracetamol')
        """
        if not query or len(query.strip()) < 2:
            return None

        clean_query = self._normalize(query)
        # Strip common prescription prefixes if present in query
        clean_query = re.sub(r'^(tab|cap|syr|inj|ointment|oint|susp|drop|drops)\s+', '', clean_query)
        
        # Stopword and non-drug filter
        if clean_query in self.STOPWORDS:
            return None
            
        # Reject purely numeric or single/two-character non-drug strings
        if len(clean_query) < 3 or clean_query.isdigit():
            return None

        q_soundex = soundex(clean_query)

        best_match = None
        best_score = 0.0
        best_matched_term = ""

        for term, term_soundex, drug in self.search_index:
            # Direct exact match
            if clean_query == term:
                return {
                    "matched": True,
                    "confidence": 100.0,
                    "matched_term": term,
                    "drug": drug,
                    "generic_name": drug["generic_name"],
                    "class": drug["class"],
                    "match_type": "exact"
                }

            # If query is short (<= 3 chars), ONLY allow exact match
            if len(clean_query) <= 3:
                continue

            # Levenshtein metrics
            sort_score = fuzz.token_sort_ratio(clean_query, term)
            ratio_score = fuzz.ratio(clean_query, term)
            partial_score = fuzz.partial_ratio(clean_query, term) if len(term) >= 4 else 0

            # Composite heuristic score
            score = max(sort_score, ratio_score * 0.95, partial_score * 0.80)

            # Doctor cursive shorthand boost: e.g. "Amox" matching "Amoxicillin" (min 4 chars)
            if len(clean_query) >= 4 and term.startswith(clean_query):
                score = max(score, 90.0)

            # Soundex phonetic bonus (only if length >= 4)
            if q_soundex == term_soundex and len(clean_query) >= 4:
                score = min(100.0, score + 12.0)

            if score > best_score:
                best_score = score
                best_match = drug
                best_matched_term = term

        if best_match and best_score >= threshold:
            return {
                "matched": True,
                "confidence": round(best_score, 1),
                "matched_term": best_matched_term,
                "drug": best_match,
                "generic_name": best_match["generic_name"],
                "class": best_match["class"],
                "match_type": "fuzzy" if best_score < 99 else "exact"
            }

        return None

    def find_close_matches(self, query: str, top_k: int = 3, threshold: float = 40.0) -> List[Dict[str, Any]]:
        """Finds top-k closest candidate matches for a messy OCR drug string.
        
        Evaluates exact matches, phonetic Soundex, Levenshtein edit distance,
        and common OCR cursive character substitutions (rn<->m, cl<->d, etc.).
        """
        if not query or len(query.strip()) < 2:
            return []

        clean_query = self._normalize(query)
        clean_query = re.sub(r'^(tab|cap|syr|inj|ointment|oint|susp|drop|drops)\s+', '', clean_query)
        if clean_query in self.STOPWORDS or clean_query.isdigit() or len(clean_query) < 3:
            return []

        q_soundex = soundex(clean_query)
        
        # OCR confusion variants
        variants = {clean_query}
        if "rn" in clean_query:
            variants.add(clean_query.replace("rn", "m"))
        if "m" in clean_query:
            variants.add(clean_query.replace("m", "rn"))
        if "n" in clean_query:
            variants.add(clean_query.replace("n", "m"))
        if "cl" in clean_query:
            variants.add(clean_query.replace("cl", "d"))
        if "d" in clean_query:
            variants.add(clean_query.replace("d", "cl"))
        if "z" in clean_query:
            variants.add(clean_query.replace("z", "g"))
        if "g" in clean_query:
            variants.add(clean_query.replace("g", "z"))
        if "s" in clean_query:
            variants.add(clean_query.replace("s", "o"))
        if "r" in clean_query:
            variants.add(clean_query.replace("r", "x"))
        if "x" in clean_query:
            variants.add(clean_query.replace("x", "r"))

        candidates = {}

        for term, term_soundex, drug in self.search_index:
            gen = drug["generic_name"]

            if clean_query == term:
                score = 100.0
                m_type = "Exact Match"
            else:
                sort_score = fuzz.token_sort_ratio(clean_query, term)
                ratio_score = fuzz.ratio(clean_query, term)
                partial_score = fuzz.partial_ratio(clean_query, term) if len(term) >= 4 else 0
                score = max(sort_score, ratio_score * 0.95, partial_score * 0.80)
                m_type = "Edit Distance"

                if len(clean_query) >= 4 and term.startswith(clean_query):
                    score = max(score, 90.0)
                    m_type = "Cursive Prefix"

                if q_soundex == term_soundex and len(clean_query) >= 4:
                    score = min(100.0, score + 15.0)
                    m_type = "Phonetic / Soundex"

                for v in variants:
                    v_ratio = fuzz.ratio(v, term)
                    if v_ratio > 70:
                        cand_s = v_ratio + 5.0
                        if cand_s > score:
                            score = min(98.0, cand_s)
                            m_type = "OCR Typo Correction"

            if score >= threshold:
                if gen not in candidates or score > candidates[gen]["confidence"]:
                    candidates[gen] = {
                        "generic_name": gen,
                        "matched_term": term.title(),
                        "confidence": round(score, 1),
                        "match_type": m_type,
                        "class": drug["class"],
                        "drug": drug
                    }

        sorted_cands = sorted(candidates.values(), key=lambda x: x["confidence"], reverse=True)
        return sorted_cands[:top_k]

    def get_all_generic_names(self) -> List[str]:
        """Returns unique list of all generic names."""
        return sorted(list(set(d["generic_name"] for d in self.drugs)))


# Singleton instance
drug_db = DrugDatabase()
