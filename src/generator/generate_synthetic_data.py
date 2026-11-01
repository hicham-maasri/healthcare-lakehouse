#!/usr/bin/env python3
"""
Synthetic multi-hospital healthcare dataset generator for the Healthcare Lakehouse project.

Simulates TWO hospital source systems (HA = large academic hospital, HB = community hospital)
that drop CSV files into the ADLS "landing" zone, in time-ordered batches, exactly the way
Auto Loader expects to receive them.

Entities (per hospital)   patients, providers, encounters, transactions (charge lines), claims
Shared reference files    hospitals, departments, icd10_codes, procedures, payors   (batch 1 only)

Design rules
  * Generated TOP-DOWN: patient -> encounters -> charge lines -> claim. Every ID is inherited,
    never re-randomised, so all joins work (unless you inject dirt on purpose).
  * Values are DERIVED, not uniform random: diagnoses depend on age/sex/season, encounter type
    depends on severity, charges depend on procedure and hospital price level, payment depends
    on payor rules, claims have a life cycle (Pending -> Paid/Denied) across batches.
  * The two hospitals differ in size, price level, payor mix, denial rate AND file conventions
    (date format, gender codes, column names, phone format) so Silver has real conformance work.
  * A controlled amount of dirt is injected (see --dirty-rate) and logged in manifest.json,
    which acts as an "answer key" for your data-quality checks.
  * ALL DATA IS FAKE. SSNs use the 9xx prefix (never issued), e-mails use example.com.
    Procedure codes are ILLUSTRATIVE (PXnnnn), not AMA CPT. ICD-10-CM codes are real/public.

Usage
    python generate_synthetic_data.py                    # 6 batches into data/batches/
    python generate_synthetic_data.py --scale 5          # 5x volume (≈ 61k patients, ≈ 145k encounters)
    python generate_synthetic_data.py --clean            # no injected dirt (for referential-integrity tests)
    python generate_synthetic_data.py --batches 4 --seed 7 --out data/batches
Standard library only.
"""
import argparse, csv, json, math, random
from collections import Counter, defaultdict, namedtuple
from datetime import date, timedelta
from pathlib import Path

START, END = date(2023, 1, 1), date(2025, 12, 31)
ENC_END = END - timedelta(days=30)          # leave room for claims to be submitted inside the window

# --------------------------------------------------------------------------- reference data
DEPARTMENTS = [
    ("DEPT001", "Emergency", "Clinical"), ("DEPT002", "Cardiology", "Clinical"), ("DEPT003", "Neurology", "Clinical"),
    ("DEPT004", "Oncology", "Clinical"), ("DEPT005", "Pediatrics", "Clinical"), ("DEPT006", "Orthopedics", "Clinical"),
    ("DEPT007", "Dermatology", "Clinical"), ("DEPT008", "Gastroenterology", "Clinical"), ("DEPT009", "Urology", "Clinical"),
    ("DEPT010", "Radiology", "Diagnostic"), ("DEPT011", "Anesthesiology", "Support"), ("DEPT012", "Pathology", "Diagnostic"),
    ("DEPT013", "Surgery", "Clinical"), ("DEPT014", "Pulmonology", "Clinical"), ("DEPT015", "Nephrology", "Clinical"),
    ("DEPT016", "Ophthalmology", "Clinical"), ("DEPT017", "Gynecology", "Clinical"), ("DEPT018", "Psychiatry", "Clinical"),
    ("DEPT019", "Endocrinology", "Clinical"), ("DEPT020", "Rheumatology", "Clinical"),
]
DEPT_ID = {n: i for i, n, _ in DEPARTMENTS}

Dx = namedtuple("Dx", "code desc cat dept sev chronic lo hi sex w")   # w = prevalence@60 (chronic) or weight (acute)
DIAGNOSES = [
    # ---- chronic conditions: w = prevalence among eligible people around age 60
    Dx("I10", "Essential (primary) hypertension", "Cardiovascular", "Cardiology", 2, True, 30, 100, None, .30),
    Dx("I50.9", "Heart failure, unspecified", "Cardiovascular", "Cardiology", 4, True, 50, 100, None, .05),
    Dx("I48.91", "Unspecified atrial fibrillation", "Cardiovascular", "Cardiology", 3, True, 50, 100, None, .06),
    Dx("E11.9", "Type 2 diabetes mellitus without complications", "Endocrine", "Endocrinology", 3, True, 30, 100, None, .14),
    Dx("E78.5", "Hyperlipidemia, unspecified", "Endocrine", "Endocrinology", 2, True, 30, 100, None, .22),
    Dx("E03.9", "Hypothyroidism, unspecified", "Endocrine", "Endocrinology", 2, True, 20, 100, None, .07),
    Dx("J45.909", "Unspecified asthma, uncomplicated", "Respiratory", "Pulmonology", 2, True, 0, 100, None, .08),
    Dx("J44.9", "Chronic obstructive pulmonary disease, unspecified", "Respiratory", "Pulmonology", 4, True, 50, 100, None, .06),
    Dx("N18.9", "Chronic kidney disease, unspecified", "Genitourinary", "Nephrology", 4, True, 45, 100, None, .06),
    Dx("K21.9", "Gastro-esophageal reflux disease without esophagitis", "Digestive", "Gastroenterology", 1, True, 20, 100, None, .12),
    Dx("M17.9", "Osteoarthritis of knee, unspecified", "Musculoskeletal", "Orthopedics", 2, True, 45, 100, None, .12),
    Dx("M81.0", "Age-related osteoporosis without current pathological fracture", "Musculoskeletal", "Rheumatology", 2, True, 55, 100, "F", .09),
    Dx("M06.9", "Rheumatoid arthritis, unspecified", "Musculoskeletal", "Rheumatology", 3, True, 30, 90, None, .03),
    Dx("F32.9", "Major depressive disorder, single episode, unspecified", "Mental health", "Psychiatry", 3, True, 15, 90, None, .09),
    Dx("F41.9", "Anxiety disorder, unspecified", "Mental health", "Psychiatry", 2, True, 15, 90, None, .10),
    Dx("G43.909", "Migraine, unspecified, not intractable, without status migrainosus", "Neurological", "Neurology", 2, True, 15, 65, None, .07),
    Dx("G40.909", "Epilepsy, unspecified, not intractable, without status epilepticus", "Neurological", "Neurology", 3, True, 0, 100, None, .015),
    Dx("C50.919", "Malignant neoplasm of unspecified site of unspecified female breast", "Oncology", "Oncology", 5, True, 35, 90, "F", .02),
    Dx("C34.90", "Malignant neoplasm of unspecified part of unspecified bronchus or lung", "Oncology", "Oncology", 5, True, 50, 90, None, .015),
    Dx("C18.9", "Malignant neoplasm of colon, unspecified", "Oncology", "Oncology", 5, True, 45, 90, None, .015),
    Dx("L20.9", "Atopic dermatitis, unspecified", "Dermatology", "Dermatology", 1, True, 0, 100, None, .05),
    Dx("L40.9", "Psoriasis, unspecified", "Dermatology", "Dermatology", 2, True, 15, 100, None, .03),
    Dx("H25.9", "Unspecified age-related cataract", "Eye", "Ophthalmology", 2, True, 55, 100, None, .10),
    # ---- acute conditions: w = relative frequency
    Dx("I21.9", "Acute myocardial infarction, unspecified", "Cardiovascular", "Cardiology", 5, False, 40, 100, None, 1.0),
    Dx("I63.9", "Cerebral infarction, unspecified", "Neurological", "Neurology", 5, False, 50, 100, None, 1.0),
    Dx("J18.9", "Pneumonia, unspecified organism", "Respiratory", "Pulmonology", 4, False, 0, 100, None, 3.0),
    Dx("J06.9", "Acute upper respiratory infection, unspecified", "Respiratory", "Pulmonology", 1, False, 0, 100, None, 6.0),
    Dx("U07.1", "COVID-19", "Respiratory", "Pulmonology", 3, False, 0, 100, None, 2.0),
    Dx("N39.0", "Urinary tract infection, site not specified", "Genitourinary", "Urology", 2, False, 5, 100, None, 3.0),
    Dx("N20.0", "Calculus of kidney", "Genitourinary", "Urology", 3, False, 25, 80, None, 1.5),
    Dx("K29.70", "Gastritis, unspecified, without bleeding", "Digestive", "Gastroenterology", 1, False, 12, 100, None, 2.0),
    Dx("K35.80", "Unspecified acute appendicitis", "Digestive", "Surgery", 4, False, 5, 60, None, 1.0),
    Dx("K80.20", "Calculus of gallbladder without cholecystitis", "Digestive", "Surgery", 3, False, 30, 85, None, 1.0),
    Dx("M54.50", "Low back pain, unspecified", "Musculoskeletal", "Orthopedics", 1, False, 18, 90, None, 4.0),
    Dx("S72.009A", "Fracture of unspecified part of neck of unspecified femur, initial encounter", "Trauma", "Orthopedics", 4, False, 60, 100, None, .6),
    Dx("S06.0X0A", "Concussion without loss of consciousness, initial encounter", "Trauma", "Emergency", 2, False, 5, 60, None, 1.5),
    Dx("T14.90XA", "Injury, unspecified, initial encounter", "Trauma", "Emergency", 2, False, 0, 100, None, 3.0),
    Dx("O80", "Encounter for full-term uncomplicated delivery", "Women's health", "Gynecology", 1, False, 18, 42, "F", 1.5),
    Dx("N94.6", "Dysmenorrhea, unspecified", "Women's health", "Gynecology", 1, False, 12, 50, "F", 1.2),
    Dx("H10.9", "Unspecified conjunctivitis", "Eye", "Ophthalmology", 1, False, 0, 100, None, 2.0),
]
CHRONIC = [d for d in DIAGNOSES if d.chronic]
ACUTE = [d for d in DIAGNOSES if not d.chronic]
DX_BY_CODE = {d.code: d for d in DIAGNOSES}
AGE_FLAT = {"J45.909", "L20.9", "F41.9", "F32.9", "G43.909", "G40.909", "L40.9"}   # prevalence not strongly age driven
WINTER_DX = {"J18.9", "J06.9", "U07.1", "J44.9", "J45.909"}

# Illustrative procedure catalogue (NOT AMA CPT):  code -> (description, department, standard charge USD)
PROCEDURES = {
    "PX0001": ("Office visit, new patient", None, 220), "PX0002": ("Office visit, established patient", None, 140),
    "PX0003": ("Telemedicine consultation", None, 95), "PX0004": ("Emergency visit, moderate severity", "Emergency", 650),
    "PX0005": ("Emergency visit, high severity", "Emergency", 1450), "PX0006": ("Inpatient room and daily care (per day)", None, 2400),
    "PX0007": ("Inpatient admission and evaluation", None, 900),
    "PX1001": ("Electrocardiogram", "Cardiology", 120), "PX1002": ("Echocardiogram", "Cardiology", 780),
    "PX1003": ("Cardiac catheterization", "Cardiology", 6800),
    "PX1101": ("Chest X-ray", "Radiology", 180), "PX1102": ("CT scan", "Radiology", 1250),
    "PX1103": ("MRI scan", "Radiology", 2100), "PX1104": ("Ultrasound", "Radiology", 520),
    "PX1201": ("Complete blood count", "Pathology", 45), "PX1202": ("Comprehensive metabolic panel", "Pathology", 85),
    "PX1203": ("Tissue biopsy and pathology", "Pathology", 760), "PX1204": ("HbA1c test", "Pathology", 55),
    "PX1301": ("Appendectomy", "Surgery", 9200), "PX1302": ("Cholecystectomy", "Surgery", 11800),
    "PX1303": ("Hip fracture repair", "Orthopedics", 15500), "PX1304": ("General anesthesia (per case)", "Anesthesiology", 1200),
    "PX1401": ("Spirometry", "Pulmonology", 150), "PX1402": ("Nebulizer treatment", "Pulmonology", 85),
    "PX1501": ("Dialysis session", "Nephrology", 950),
    "PX1601": ("Cataract surgery", "Ophthalmology", 4800), "PX1602": ("Comprehensive eye exam", "Ophthalmology", 210),
    "PX1701": ("Vaginal delivery", "Gynecology", 7200), "PX1801": ("Psychotherapy session", "Psychiatry", 190),
    "PX1901": ("Diabetes education session", "Endocrinology", 110),
    "PX2001": ("Joint injection", "Orthopedics", 340), "PX2002": ("Physical therapy session", "Orthopedics", 130),
    "PX2101": ("Chemotherapy infusion", "Oncology", 3600), "PX2102": ("Radiation therapy session", "Oncology", 1900),
    "PX2201": ("Skin biopsy", "Dermatology", 310), "PX2202": ("Phototherapy session", "Dermatology", 140),
    "PX2301": ("Electroencephalogram (EEG)", "Neurology", 640), "PX2302": ("Acute stroke intervention", "Neurology", 5200),
    "PX2401": ("Endoscopy", "Gastroenterology", 1700), "PX2501": ("Urinalysis", "Pathology", 30),
    "PX2502": ("Lithotripsy", "Urology", 4200), "PX2601": ("Wound care and suturing", "Emergency", 380),
    "PX2602": ("Fracture splinting", "Emergency", 290),
}
CATEGORY_PROCS = {
    "Cardiovascular": ["PX1001", "PX1002", "PX1202", "PX1101"], "Endocrine": ["PX1204", "PX1202", "PX1901"],
    "Respiratory": ["PX1101", "PX1401", "PX1402", "PX1201", "PX1102"], "Genitourinary": ["PX2501", "PX1104", "PX1501", "PX1202"],
    "Digestive": ["PX2401", "PX1104", "PX1201"], "Musculoskeletal": ["PX2001", "PX2002", "PX1103", "PX1101"],
    "Trauma": ["PX2601", "PX2602", "PX1101", "PX1102"], "Neurological": ["PX2301", "PX1103", "PX1102"],
    "Oncology": ["PX2101", "PX2102", "PX1203", "PX1102"], "Mental health": ["PX1801"],
    "Women's health": ["PX1104", "PX1201"], "Dermatology": ["PX2201", "PX2202"], "Eye": ["PX1602"],
}
KEY_PROC = {"K35.80": "PX1301", "K80.20": "PX1302", "S72.009A": "PX1303", "H25.9": "PX1601",
            "O80": "PX1701", "I21.9": "PX1003", "I63.9": "PX2302"}
NEEDS_ANESTHESIA = {"K35.80", "K80.20", "S72.009A", "H25.9"}

Payor = namedtuple("Payor", "id name type lob allowed denial days")
PAYORS = {p.id: p for p in [
    Payor("PAY01", "Medicare", "Government", "Medicare", .38, .06, 35),
    Payor("PAY02", "Medicaid", "Government", "Medicaid", .30, .09, 50),
    Payor("PAY03", "BlueCross", "Private", "Commercial", .55, .08, 28),
    Payor("PAY04", "Aetna", "Private", "Commercial", .52, .10, 30),
    Payor("PAY05", "UnitedHealthcare", "Private", "Commercial", .50, .12, 32),
    Payor("PAY06", "Self-Pay", "Self-pay", "Self-Pay", .35, .00, 70),
]}
DENIAL_REASONS = ["Not medically necessary", "Prior authorization missing", "Duplicate claim", "Coverage terminated",
                  "Out of network", "Timely filing limit exceeded", "Coding error"]

HOSPITALS = {
    "HA": dict(name="Riverside University Hospital", city="Chicago", state="IL", type="Academic Medical Center", beds=640,
               patients=8000, providers=60, price=1.25, denial=1.0, age_shift=4,
               mix={"Inpatient": 1.3, "Outpatient": .9, "Emergency": 1.0},
               payor_u65={"PAY03": .22, "PAY04": .15, "PAY05": .15, "PAY02": .28, "PAY06": .20},
               cities=[("Chicago", "IL", "606"), ("Evanston", "IL", "602"), ("Oak Park", "IL", "603"), ("Naperville", "IL", "605")],
               date_fmt="%Y-%m-%d", gender={"M": "Male", "F": "Female"}, dob_col="DOB", phone="({a}) {b}-{c}"),
    "HB": dict(name="Lakeside Community Medical Center", city="Columbus", state="OH", type="Community Hospital", beds=180,
               patients=4000, providers=30, price=0.90, denial=1.30, age_shift=0,
               mix={"Inpatient": .7, "Outpatient": 1.2, "Emergency": 1.1},
               payor_u65={"PAY03": .20, "PAY04": .12, "PAY05": .12, "PAY02": .30, "PAY06": .26},
               cities=[("Columbus", "OH", "432"), ("Dublin", "OH", "430"), ("Westerville", "OH", "430"), ("Delaware", "OH", "430")],
               date_fmt="%m/%d/%Y", gender={"M": "M", "F": "F"}, dob_col="DateOfBirth", phone="{a}-{b}-{c}"),
}

FIRST_M = "James John Robert Michael William David Richard Joseph Thomas Daniel Matthew Anthony Mark Steven Paul Andrew Kevin Brian George".split()
FIRST_F = "Mary Patricia Jennifer Linda Elizabeth Barbara Susan Jessica Sarah Karen Nancy Lisa Margaret Betty Sandra Ashley Emily Donna Michelle Carol".split()
LAST = "Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez Hernandez Lopez Wilson Anderson Thomas Taylor Moore Jackson Martin Lee Perez Thompson White Harris Clark Lewis Robinson Walker Young Allen".split()
STREETS = "Main Oak Maple Cedar Elm Washington Lake Hill Park Pine Sunset Highland Church Mill River".split()
SUFFIX = ["St", "Ave", "Rd", "Blvd", "Ln", "Dr"]

# --------------------------------------------------------------------------- helpers
def poisson(rng, lam):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        k += 1; p *= rng.random()
        if p <= L:
            return k - 1

def rand_date(rng, a, b):
    return a + timedelta(days=rng.randint(0, max(0, (b - a).days)))

def age_at(dob, d):
    return (d - dob).days // 365

def eligible(dx, age, sex):
    return dx.lo <= age <= dx.hi and (dx.sex is None or dx.sex == sex)

def weighted(rng, items, weights):
    return rng.choices(items, weights=weights, k=1)[0]

def make_address(rng, h):
    city, state, zip3 = rng.choice(h["cities"])
    return f"{rng.randint(10, 9899)} {rng.choice(STREETS)} {rng.choice(SUFFIX)}", city, state, f"{zip3}{rng.randint(0, 99):02d}"

def make_phone(rng, h):
    return h["phone"].format(a=rng.choice(["312", "773", "614", "708", "440"]), b=rng.randint(200, 999), c=f"{rng.randint(0, 9999):04d}")

# --------------------------------------------------------------------------- world model
def pick_chronic(rng, age, sex):
    f = max(0.05, min(1.6, (age - 20) / 40))
    got = []
    for dx in CHRONIC:
        if not eligible(dx, age, sex):
            continue
        p = dx.w * (0.6 if dx.code in AGE_FLAT else f)
        if rng.random() < p:
            got.append(dx.code)
    if "E11.9" in got and "I10" not in got and rng.random() < .35:
        got.append("I10")
    rng.shuffle(got)
    return got[:3]

def pick_type(rng, h, dx, chronic_flag):
    if dx.code == "O80":
        return "Inpatient"
    if chronic_flag:
        if dx.sev >= 4 and rng.random() < .12:
            return "Inpatient"
        w = {"Routine Checkup": 40, "Telemedicine": 25, "Outpatient": 35}
    elif dx.sev >= 4:
        w = {"Inpatient": 55, "Emergency": 45}
    elif dx.sev >= 2:
        w = {"Emergency": 35, "Outpatient": 50, "Telemedicine": 15}
    else:
        w = {"Outpatient": 60, "Telemedicine": 25, "Emergency": 15}
    types = list(w)
    return weighted(rng, types, [w[t] * h["mix"].get(t, 1.0) for t in types])

def build_hospital(rng, hid, scale, nb):
    h = HOSPITALS[hid]
    span = ((END - START).days + 1) / nb
    def batch_of(d):
        return 1 if d < START else min(nb, 1 + int((d - START).days // span))

    # ---- providers: at least one per department, rest weighted towards big departments
    providers, by_dept = [], defaultdict(list)
    n_prov = max(20, int(h["providers"] * max(1, scale ** .5)))
    dept_names = [n for _, n, _ in DEPARTMENTS]
    assign = dept_names + rng.choices(dept_names, weights=[3 if n in ("Emergency", "Surgery", "Cardiology") else 1 for n in dept_names], k=n_prov - len(dept_names))
    npi_used = set()
    for i, dept in enumerate(assign, 1):
        while True:
            npi = "1" + "".join(str(rng.randint(0, 9)) for _ in range(9))
            if npi not in npi_used:
                npi_used.add(npi); break
        pid = f"{hid}-PRV{i:04d}"
        hire = rand_date(rng, date(2005, 1, 1), date(2022, 12, 31))
        providers.append(dict(ProviderID=pid, FirstName=rng.choice(FIRST_M + FIRST_F), LastName=rng.choice(LAST),
                              Specialization=dept, DeptID=DEPT_ID[dept], NPI=npi, HireDate=hire,
                              InsertDate=START - timedelta(days=30), ModifiedDate=START - timedelta(days=30), _batch=1))
        by_dept[dept].append(pid)

    patients, encounters, charges, claims = [], [], [], []
    ctr = Counter()
    n_pat = int(h["patients"] * scale)
    total_days = (ENC_END - START).days
    u65 = list(h["payor_u65"]); u65w = [h["payor_u65"][k] for k in u65]

    for pi in range(1, n_pat + 1):
        sex = rng.choice("MF")
        age = int(min(95, max(0, rng.gauss(46 + h["age_shift"], 22))))
        dob = START - timedelta(days=age * 365 + rng.randint(0, 364))
        first = rng.choice(FIRST_M if sex == "M" else FIRST_F); last = rng.choice(LAST)
        chronic = pick_chronic(rng, age, sex)
        if age >= 65:
            payor = "PAY01" if rng.random() < .88 else weighted(rng, u65, u65w)
        else:
            payor = weighted(rng, u65, u65w)
        existing = rng.random() < .70
        reg = rand_date(rng, date(2012, 1, 1), START - timedelta(days=1)) if existing else rand_date(rng, START, ENC_END - timedelta(days=20))
        s = START if existing else reg
        lam = (1.0 + 1.4 * len(chronic) + (0.6 if age >= 65 else 0.0)) * ((ENC_END - s).days / total_days)
        n_enc = poisson(rng, lam)
        if not existing:
            n_enc = max(1, n_enc)
        dates = sorted(rand_date(rng, s, ENC_END) for _ in range(n_enc))
        if not existing and dates:
            dates[0] = reg
        pid = f"{hid}-P{pi:06d}"
        street, city, state, zipc = make_address(rng, h)
        p = {"PatientID": pid, "FirstName": first, "LastName": last, "MiddleName": rng.choice("ABCDEFGHJKLMNPRSTW"),
             "Gender": h["gender"][sex], h["dob_col"]: dob,
             "SSN": f"9{rng.randint(0, 99):02d}-{rng.randint(1, 99):02d}-{rng.randint(1, 9999):04d}",
             "PhoneNumber": make_phone(rng, h), "Email": f"{first}.{last}{rng.randint(1, 999)}@example.com".lower(),
             "Address": street, "City": city, "State": state, "ZipCode": zipc, "PayorID": payor,
             "InsertDate": reg, "ModifiedDate": reg, "_batch": batch_of(reg)}
        patients.append(p)
        # ~3% of patients move / change phone later -> an updated version arrives in a later batch
        if rng.random() < .03:
            upd = rand_date(rng, max(reg, START) + timedelta(days=30), END)
            street2, city2, state2, zip2 = make_address(rng, h)
            u = dict(p, Address=street2, City=city2, State=state2, ZipCode=zip2, PhoneNumber=make_phone(rng, h),
                     ModifiedDate=upd, _batch=batch_of(upd))
            u["_is_update"] = True
            patients.append(u)

        seen_first = False
        for d in dates:
            a = age_at(dob, d)
            is_chronic = bool(chronic) and rng.random() < .6
            if is_chronic:
                dx = DX_BY_CODE[rng.choice(chronic)]
            else:
                pool = [x for x in ACUTE if eligible(x, a, sex)]
                season = [(1.8 if d.month in (12, 1, 2, 3) else .8) if x.code in WINTER_DX else 1.0 for x in pool]
                dx = weighted(rng, pool, [x.w * m for x, m in zip(pool, season)])
            etype = pick_type(rng, h, dx, is_chronic)
            dept = "Emergency" if etype == "Emergency" else dx.dept
            if a < 16 and dept not in ("Emergency",):
                dept = "Pediatrics"
            prov = rng.choice(by_dept[dept])
            if etype == "Inpatient":
                los = rng.randint(1, 3) if dx.code == "O80" else max(1, int(rng.lognormvariate(math.log(1.5 + dx.sev * .9), .45)))
            else:
                los = 0
            disch = d + timedelta(days=los)
            em = {"Inpatient": "PX0007", "Emergency": "PX0004" if dx.sev <= 2 else "PX0005",
                  "Outpatient": "PX0002" if seen_first else "PX0001", "Routine Checkup": "PX0002", "Telemedicine": "PX0003"}[etype]
            seen_first = True
            ctr["enc"] += 1
            eid = f"{hid}-E{ctr['enc']:07d}"
            ins = disch + timedelta(days=rng.randint(0, 2))
            encounters.append(dict(EncounterID=eid, PatientID=pid, ProviderID=prov, DeptID=DEPT_ID[dept], EncounterDate=d,
                                   DischargeDate=disch, EncounterType=etype, PrimaryICD10=dx.code, ProcedureCode=em,
                                   InsertDate=ins, ModifiedDate=ins, _batch=batch_of(ins)))
            # ---- charge lines
            lines = [("PX0007", 1), ("PX0006", los)] if etype == "Inpatient" else [(em, 1)]
            n_anc = {"Routine Checkup": rng.randint(0, 1), "Telemedicine": rng.randint(0, 1), "Outpatient": rng.randint(0, 2),
                     "Emergency": rng.randint(1, 3), "Inpatient": rng.randint(2, 4)}[etype]
            pool = [c for c in CATEGORY_PROCS.get(dx.cat, []) if c not in KEY_PROC.values()]
            lines += [(c, 1) for c in rng.sample(pool, min(n_anc, len(pool)))]
            if dx.code in KEY_PROC and rng.random() < .9:
                lines.append((KEY_PROC[dx.code], 1))
                if dx.code in NEEDS_ANESTHESIA:
                    lines.append(("PX1304", 1))
            ctr["clm"] += 1
            cid = f"{hid}-C{ctr['clm']:07d}"
            total, last_post = 0.0, disch
            for code, units in lines:
                if units <= 0:
                    continue
                desc, pdept, price = PROCEDURES[code]
                ldept = pdept or dept
                lprov = prov if ldept == dept else rng.choice(by_dept[ldept])
                sdate = d + timedelta(days=rng.randint(0, los)) if etype == "Inpatient" and code not in ("PX0007",) else d
                amount = round(price * units * h["price"] * rng.lognormvariate(0, .10), 2)
                posted = sdate + timedelta(days=rng.randint(0, 3))
                last_post = max(last_post, posted)
                total += amount
                ctr["trx"] += 1
                charges.append(dict(TransactionID=f"{hid}-T{ctr['trx']:07d}", EncounterID=eid, ClaimID=cid, PatientID=pid,
                                    ProviderID=lprov, DeptID=DEPT_ID[ldept], ServiceDate=sdate, PostedDate=posted,
                                    ProcedureCode=code, ICD10Code=dx.code, Units=units, Amount=amount,
                                    InsertDate=posted, ModifiedDate=posted, _batch=batch_of(posted)))
            # ---- claim (one per encounter) with a Pending -> final life cycle
            total = round(total, 2)
            py = PAYORS[payor]
            claim_date = last_post + timedelta(days=rng.randint(1, 5))
            adj_date = claim_date + timedelta(days=max(7, int(rng.gauss(py.days, py.days * .3))))
            r = rng.random(); den = py.denial * h["denial"]
            if payor == "PAY06":
                status = "Paid" if r < .65 else "Denied"
                reason = None if status == "Paid" else "Uncollectible - written off"
            elif r < .015:
                status, reason = "Rejected", "Missing or invalid claim data"
            elif r < .015 + den:
                status, reason = "Denied", rng.choice(DENIAL_REASONS)
            else:
                status, reason = "Paid", None
            allowed = copay = ded = coins = paid = 0.0
            if status == "Paid":
                allowed = round(total * py.allowed * rng.uniform(.9, 1.1), 2)
                if payor == "PAY06":
                    paid = round(allowed * rng.uniform(.1, .7), 2)
                else:
                    copay = {"Outpatient": rng.choice([25, 35, 50]), "Routine Checkup": rng.choice([0, 20, 30]),
                             "Telemedicine": rng.choice([0, 15]), "Emergency": rng.choice([100, 150, 250]), "Inpatient": 0}[etype]
                    if payor == "PAY02":
                        copay = rng.choice([0, 3])
                    if py.lob == "Commercial" and rng.random() < .25:
                        ded = round(min(rng.choice([250, 500, 1000]), allowed * .5), 2)
                    rate = {"Medicare": .20, "Medicaid": 0.0}.get(py.lob, rng.uniform(.10, .20))
                    copay = min(copay, allowed)
                    coins = round(max(0.0, allowed - copay - ded) * rate, 2)
                    paid = round(max(0.0, allowed - copay - ded - coins), 2)
            base = dict(ClaimID=cid, EncounterID=eid, PatientID=pid, PayorID=payor, ServiceDate=d, ClaimDate=claim_date,
                        ClaimAmount=total, InsertDate=claim_date)
            v1 = dict(base, AllowedAmount=None, PaidAmount=None, Copay=None, Deductible=None, Coinsurance=None,
                      ClaimStatus="Pending", DenialReason=None, PaidDate=None, ModifiedDate=claim_date, _batch=batch_of(claim_date))
            v2 = dict(base, AllowedAmount=allowed, PaidAmount=paid, Copay=copay, Deductible=ded, Coinsurance=coins,
                      ClaimStatus=status, DenialReason=reason, PaidDate=adj_date, ModifiedDate=adj_date, _batch=batch_of(adj_date))
            if adj_date > END:
                claims.append(v1)                                   # still pending when the data ends
            elif v1["_batch"] == v2["_batch"]:
                claims.append(v2)
            else:
                claims.extend([v1, v2])
    # patient: if an update lands in the same batch as the first version, keep only the update
    keep, by_id = [], defaultdict(list)
    for p in patients:
        by_id[p["PatientID"]].append(p)
    for versions in by_id.values():
        if len(versions) == 2 and versions[0]["_batch"] == versions[1]["_batch"]:
            keep.append(versions[1])
        else:
            keep.extend(versions)
    return dict(providers=providers, patients=keep, encounters=encounters, transactions=charges, claims=claims)

# --------------------------------------------------------------------------- dirt injection
def apply_dirt(entity, rows, hid, rng, rate, stats):
    if rate <= 0:
        return rows
    h = HOSPITALS[hid]
    def hit(w=1.0):
        return rng.random() < rate * w
    def mark(k):
        stats[f"{entity}.{k}"] += 1
    out = []
    for r0 in rows:
        r = dict(r0)
        if entity == "patients":
            if hit():
                r["FirstName"] = f" {r['FirstName'].upper()} "; r["LastName"] = r["LastName"].lower(); mark("name_case_whitespace")
            if hit(.75):
                r[h["dob_col"]] = None; mark("missing_dob")
            if hit():
                r["Gender"] = rng.choice(["M", "F", "male", "FEMALE", ""] if hid == "HA" else ["Male", "f", ""]); mark("gender_variant")
            if hit():
                r["PhoneNumber"] = None; mark("missing_phone")
            if hit(.5):
                r["Email"] = r["Email"].replace("@", ""); mark("invalid_email")
        elif entity == "providers":
            if hit(1.5):
                r["NPI"] = r["NPI"][:8]; mark("short_npi")
        elif entity == "encounters":
            if hit(.25):
                r["PatientID"] = f"{hid}-P999999"; mark("orphan_patient")
            if hit(.25):
                r["DischargeDate"] = r["EncounterDate"] - timedelta(days=rng.randint(1, 3)); mark("discharge_before_admission")
            if hit(.5):
                r["PrimaryICD10"] = rng.choice([r["PrimaryICD10"].lower(), r["PrimaryICD10"].replace(".", ""), ""]); mark("malformed_icd10")
            if hit():
                r["EncounterType"] = rng.choice([f" {r['EncounterType'].lower()}", r["EncounterType"].upper()]); mark("encounter_type_casing")
            if hit(.25):
                r["ModifiedDate"] = r["InsertDate"] - timedelta(days=rng.randint(1, 5)); mark("modified_before_inserted")
            if hit(.5):
                alt = "%m/%d/%Y" if h["date_fmt"] == "%Y-%m-%d" else "%Y-%m-%d"
                r["EncounterDate"] = r["EncounterDate"].strftime(alt); mark("mixed_date_format")
        elif entity == "transactions":
            if hit(.25):
                r["Amount"] = -abs(r["Amount"]); mark("negative_amount")
            if hit(.25):
                r["Amount"] = None; mark("missing_amount")
            if hit(.15):
                r["EncounterID"] = f"{hid}-E9999999"; mark("orphan_encounter")
            if hit(.25):
                r["PostedDate"] = r["ServiceDate"] - timedelta(days=rng.randint(1, 4)); mark("posted_before_service")
        elif entity == "claims":
            if hit(.25) and r["PaidAmount"] is not None:
                r["PaidAmount"] = round(r["ClaimAmount"] * rng.uniform(1.05, 1.5), 2); mark("paid_exceeds_billed")
            if hit(.15) and r["Copay"] is not None:
                r["Copay"] = -abs(r["Copay"]); mark("negative_copay")
            if hit():
                r["ClaimStatus"] = rng.choice([r["ClaimStatus"].lower(), "APPROVED", r["ClaimStatus"] + " "]); mark("status_variant")
            if hit(.25) and r["PaidDate"] is not None:
                r["PaidDate"] = r["ClaimDate"] - timedelta(days=rng.randint(1, 5)); mark("paid_before_claim")
        out.append(r)
        if entity != "providers" and hit(.5):          # exact duplicate rows
            out.append(dict(r)); mark("duplicate_row")
    return out

# --------------------------------------------------------------------------- writing
HEADERS = {
    "patients": lambda h: ["PatientID", "FirstName", "LastName", "MiddleName", "Gender", h["dob_col"], "SSN", "PhoneNumber", "Email",
                           "Address", "City", "State", "ZipCode", "PayorID", "InsertDate", "ModifiedDate"],
    "providers": lambda h: ["ProviderID", "FirstName", "LastName", "Specialization", "DeptID", "NPI", "HireDate", "InsertDate", "ModifiedDate"],
    "encounters": lambda h: ["EncounterID", "PatientID", "ProviderID", "DeptID", "EncounterDate", "DischargeDate", "EncounterType",
                             "PrimaryICD10", "ProcedureCode", "InsertDate", "ModifiedDate"],
    "transactions": lambda h: ["TransactionID", "EncounterID", "ClaimID", "PatientID", "ProviderID", "DeptID", "ServiceDate", "PostedDate",
                               "ProcedureCode", "ICD10Code", "Units", "Amount", "InsertDate", "ModifiedDate"],
    "claims": lambda h: ["ClaimID", "EncounterID", "PatientID", "PayorID", "ServiceDate", "ClaimDate", "ClaimAmount", "AllowedAmount",
                         "PaidAmount", "Copay", "Deductible", "Coinsurance", "ClaimStatus", "DenialReason", "PaidDate", "InsertDate", "ModifiedDate"],
}

def to_str(v, fmt):
    if v is None:
        return ""
    if isinstance(v, date):
        return v.strftime(fmt)
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)

def write_csv(path, header, rows, fmt="%Y-%m-%d"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([to_str(r.get(c), fmt) for c in header])

def write_reference(out):
    ref = out / "batch_01" / "reference"
    write_csv(ref / "hospitals.csv", ["HospitalID", "HospitalName", "City", "State", "HospitalType", "BedCapacity"],
              [dict(HospitalID=k, HospitalName=v["name"], City=v["city"], State=v["state"], HospitalType=v["type"], BedCapacity=v["beds"]) for k, v in HOSPITALS.items()])
    write_csv(ref / "departments.csv", ["DeptID", "DeptName", "DeptType"], [dict(DeptID=i, DeptName=n, DeptType=t) for i, n, t in DEPARTMENTS])
    write_csv(ref / "icd10_codes.csv", ["ICD10Code", "Description", "Category", "SeverityLevel", "ChronicFlag"],
              [dict(ICD10Code=d.code, Description=d.desc, Category=d.cat, SeverityLevel=d.sev, ChronicFlag=int(d.chronic)) for d in DIAGNOSES])
    write_csv(ref / "procedures.csv", ["ProcedureCode", "Description", "Department", "StandardCharge"],
              [dict(ProcedureCode=c, Description=v[0], Department=v[1] or "(encounter department)", StandardCharge=float(v[2])) for c, v in PROCEDURES.items()])
    write_csv(ref / "payors.csv", ["PayorID", "PayorName", "PayorType", "LineOfBusiness"],
              [dict(PayorID=p.id, PayorName=p.name, PayorType=p.type, LineOfBusiness=p.lob) for p in PAYORS.values()])

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=1.0, help="volume multiplier (1 = 12k patients, ~30k encounters)")
    ap.add_argument("--batches", type=int, default=6, help="number of time-ordered file drops")
    ap.add_argument("--dirty-rate", type=float, default=0.02, help="base probability of injected dirt per row (0 = none)")
    ap.add_argument("--clean", action="store_true", help="shortcut for --dirty-rate 0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/batches")
    a = ap.parse_args()
    rate = 0.0 if a.clean else a.dirty_rate
    out = Path(a.out)
    rng = random.Random(a.seed)
    stats, counts = Counter(), defaultdict(int)
    write_reference(out)
    for hid in HOSPITALS:
        world = build_hospital(rng, hid, a.scale, a.batches)
        fmt = HOSPITALS[hid]["date_fmt"]
        for entity, rows in world.items():
            for k in range(1, a.batches + 1):
                part = [r for r in rows if r["_batch"] == k]
                if not part:
                    continue
                part = apply_dirt(entity, part, hid, rng, rate, stats)
                rng.shuffle(part)
                write_csv(out / f"batch_{k:02d}" / entity / hid / f"{entity}_{hid}_batch{k:02d}.csv", HEADERS[entity](HOSPITALS[hid]), part, fmt)
                counts[f"{entity}.{hid}"] += len(part)
    manifest = dict(seed=a.seed, scale=a.scale, batches=a.batches, dirty_rate=rate, window=[START.isoformat(), END.isoformat()],
                    rows_written=dict(sorted(counts.items())), injected_issues=dict(sorted(stats.items())),
                    note="injected_issues is the answer key: your Silver rules should detect/fix/quarantine these. Rows from different batches that share a key are updates (MERGE).")
    (out.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {a.batches} batches to {out}/  (manifest: {out.parent / 'manifest.json'})")
    for k, v in sorted(counts.items()):
        print(f"  {k:<20}{v:>9,} rows")

if __name__ == "__main__":
    main()
