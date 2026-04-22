"""
generate_new_topic_data.py
──────────────────────────
Generates a mixed dataset (~500 records) combining:
  - Old (known) topics sampled from the labeled dataset
  - New (emerging) topics not seen during training

This simulates the real-world scenario of daily complaint ingestion where
novel topics begin to appear. The system must detect these.

Usage (PyCharm):  Run this file directly, or via terminal:
    python src/data_generation/generate_new_topic_data.py

Output:
    data/complaints_with_new_topics.csv
"""

import random
import csv
import os
from pathlib import Path

import yaml
from faker import Faker

fake = Faker()
random.seed(99)

# ─── Load Config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

LABELED_PATH = ROOT / CONFIG["data"]["labeled_dataset"]
OUTPUT_PATH = ROOT / CONFIG["data"]["new_topic_dataset"]

# ─── New / Emerging Topic Templates ───────────────────────────────────────────
NEW_TOPIC_TEMPLATES = {
    "biometric_auth_failure": [
        "My face recognition login stopped working after the latest bank app update. It keeps saying 'Face not recognized'.",
        "I registered my fingerprint for banking authentication but it no longer works. The app rejects my fingerprint every time.",
        "Biometric login on your app is completely broken on my {device}. Face ID was working perfectly before.",
        "I can't use fingerprint authentication to approve payments anymore. The sensor keeps timing out.",
        "Your app's face recognition feature is not working in low light. I have to type my password every time.",
        "After re-registering my face 5 times, the biometric login still fails consistently.",
        "The banking app keeps prompting me to register biometrics even though I already did it.",
        "I enabled facial recognition for my banking app but it keeps reverting to password login.",
        "Biometric authentication stopped working on my phone after I changed my screen protector.",
        "The fingerprint scanner in the banking app doesn't respond when I place my finger on it.",
        "Face unlock for banking is always showing 'Verification failed'. My face is the same!",
        "I enrolled my fingerprint for mobile banking but the system says it's not recognized.",
        "Cannot use Touch ID to log into the banking app since the last software update.",
        "Banking app biometric verification fails every single attempt. Forces me to use password.",
        "Fingerprint banking login worked yesterday but is completely broken today without explanation.",
    ],
    "virtual_card_issue": [
        "I generated a virtual card for online shopping but it was declined immediately.",
        "My virtual debit card number has changed without my approval and now my subscriptions are failing.",
        "I can't find my virtual card details in the banking app. The option seems to have disappeared.",
        "Virtual card generated through your app is not working on international shopping sites.",
        "I'm unable to add my virtual card to Google Pay. The verification keeps failing.",
        "My virtual credit card limit doesn't match what I set when I created it.",
        "Virtual card transactions are not showing up in my transaction history.",
        "I set spending limits on my virtual card but the limit is being ignored.",
        "The virtual card I created expired before I could use it. No warning was given.",
        "My virtual card was blocked after one declined transaction online.",
        "I cannot delete an old virtual card in the app. The delete button is not working.",
        "Virtual card CVV changes every hour but the new CVV is not being shown in the app.",
        "I'm getting charged for virtual card transactions I never made.",
        "Cannot create a new virtual card — the app says I've reached the limit but I only have one.",
        "Virtual card details are not loading in the app. The section just shows a blank screen.",
    ],
    "bnpl_payment_dispute": [
        "I signed up for buy-now-pay-later through your bank but my installment amount is incorrect.",
        "My BNPL payment was rejected even though I have sufficient balance in my account.",
        "I cancelled a BNPL purchase but I'm still being charged monthly installments.",
        "My BNPL credit limit was reduced without any notification, causing my recent purchase to fail.",
        "I made a BNPL purchase but it doesn't appear in my loan or installment section.",
        "I'm being charged interest on my BNPL plan even though it was advertised as interest-free.",
        "I paid off my BNPL balance early but the account still shows outstanding amount.",
        "I tried to enroll a purchase in BNPL but the option was greyed out in the app.",
        "My BNPL plan was supposed to be 12 months but it's charging me as if it's 6 months.",
        "I received a late payment fee for my BNPL even though I set up auto-debit.",
        "I can't view my BNPL installment schedule in the mobile banking app.",
        "The BNPL feature approved my purchase but the merchant didn't receive payment.",
        "My bank's buy-now-pay-later shows a due amount that doesn't match my purchase total.",
        "I was not informed about the processing fee when I signed up for BNPL.",
        "I want to cancel my BNPL plan but there's no cancellation option in the app.",
    ],
    "qr_payment_failure": [
        "I scanned a QR code to pay at {merchant} but the payment failed and money was deducted.",
        "QR payment via your banking app keeps saying 'Invalid QR code' even for merchants I've paid before.",
        "I paid via QR code but the merchant's system shows payment not received while my account was debited.",
        "The QR scanner in your app doesn't open. I tap the icon and nothing happens.",
        "My QR payment was successful but I received no receipt and merchant got no notification.",
        "QR payment at {merchant} failed halfway through. Money was deducted but I have no transaction record.",
        "I'm unable to generate my own QR code to receive payments via your app.",
        "QR payment limit is too low — I can't use it for larger purchases even with sufficient balance.",
        "The QR code shown in your app is blurry and merchants cannot scan it.",
        "QR payment transactions are not appearing in my statement even after 24 hours.",
        "I tried to pay via QR code and got error 'QR session expired' every time I scan.",
        "QR payments work inconsistently — sometimes succeed, sometimes fail for no apparent reason.",
        "My QR payment was reversed without explanation and my balance wasn't returned immediately.",
        "I cannot enable QR payments in settings. The toggle switches on but reverts immediately.",
        "QR code payment is extremely slow — takes over 2 minutes to process at the checkout.",
    ],
    "crypto_wallet_error": [
        "I tried to transfer crypto from my bank's digital wallet but the transaction failed.",
        "My bank's crypto wallet balance is showing incorrect amounts after a recent transfer.",
        "I cannot convert my crypto to cash through your bank platform. The option is grayed out.",
        "My Bitcoin purchase through the bank app was deducted from my account but no crypto was received.",
        "The bank's crypto feature keeps showing 'Service unavailable' when I try to access my wallet.",
        "I tried to send crypto to an external wallet but the transfer has been stuck 'pending' for 3 days.",
        "My crypto wallet in the bank app disappeared after the latest update.",
        "I'm being charged excessive fees for crypto transactions that weren't disclosed upfront.",
        "My crypto portfolio value is not updating in real time. Prices are hours behind.",
        "I cannot withdraw crypto to my hardware wallet. The destination address field rejects valid addresses.",
        "The bank's crypto exchange rate is way off market rate — I'm losing money on every conversion.",
        "I tried to buy Ethereum through your app and got 'Transaction rejected' with no reason given.",
        "My bank's digital asset account was locked without any explanation or notice.",
        "I set a crypto limit order but it was never executed even though the price target was reached.",
        "I sold my crypto holdings but the proceeds haven't been credited to my bank account after 5 days.",
    ],
}

BANKS = ["MegaBank", "FirstCity Bank", "TrustBank", "PrimeBank", "NexBank", "Unified Bank", "CitiCore Bank"]
DEVICES = ["iPhone 15", "Samsung Galaxy S24", "iPhone 14 Pro", "Pixel 8", "Huawei P60", "OnePlus 12"]
MERCHANTS = ["McDonald's", "Grab", "Shopee", "Lazada", "IKEA", "Starbucks", "Shell", "Watsons"]


def fill_template(template: str) -> str:
    return (
        template
        .replace("{device}", random.choice(DEVICES))
        .replace("{bank}", random.choice(BANKS))
        .replace("{merchant}", random.choice(MERCHANTS))
        .replace("{amount}", f"RM {random.randint(50, 5000):,}.{random.randint(0, 99):02d}")
        .replace("{ref}", fake.bothify(text="TXN-????-######").upper())
    )


def add_variation(text: str) -> str:
    prefixes = [
        "I want to report a problem: ",
        "I'm experiencing an issue where ",
        "Urgent complaint: ",
        "I need help because ",
        "",
        "",
        "",
    ]
    suffixes = [
        " Please look into this immediately.",
        " I'm very disappointed.",
        " This is urgently needed.",
        "",
        "",
    ]
    prefix = random.choice(prefixes)
    suffix = random.choice(suffixes)
    if prefix:
        text = prefix + text[0].lower() + text[1:]
    return text + suffix


def load_old_topic_samples(n_per_topic: int = 17) -> list[dict]:
    """Load a sample of known-topic complaints from the labeled dataset."""
    from collections import defaultdict
    if not LABELED_PATH.exists():
        raise FileNotFoundError(
            f"Labeled dataset not found at {LABELED_PATH}.\n"
            "Run generate_dataset.py first."
        )

    by_topic: dict[str, list[str]] = defaultdict(list)
    with open(LABELED_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_topic[row["topic"]].append(row["complaint_text"])

    samples = []
    for topic, texts in by_topic.items():
        selected = random.sample(texts, min(n_per_topic, len(texts)))
        for text in selected:
            samples.append({"complaint_text": text, "topic": topic, "is_new_topic": False})
    return samples


def generate_new_topic_records(n_per_topic: int = 60) -> list[dict]:
    """Generate complaints for the 5 new emerging topics."""
    records = []
    for topic, templates in NEW_TOPIC_TEMPLATES.items():
        count = 0
        while count < n_per_topic:
            template = random.choice(templates)
            text = fill_template(template)
            text = add_variation(text)
            records.append({"complaint_text": text, "topic": topic, "is_new_topic": True})
            count += 1
    return records


def main():
    print("🔄  Loading old topic samples...")
    old_records = load_old_topic_samples(n_per_topic=17)
    print(f"   ✅  {len(old_records)} records from {len(set(r['topic'] for r in old_records))} known topics")

    print("🔄  Generating new topic records...")
    new_records = generate_new_topic_records(n_per_topic=60)
    print(f"   ✅  {len(new_records)} records for {len(NEW_TOPIC_TEMPLATES)} new topics")

    all_records = old_records + new_records
    random.shuffle(all_records)
    print(f"\n📦  Total mixed records: {len(all_records)}")

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["complaint_text", "topic", "is_new_topic"])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"💾  Saved to: {OUTPUT_PATH}")

    # ── Distribution Summary ───────────────────────────────────────────────────
    from collections import Counter
    counts = Counter(r["topic"] for r in all_records)
    new_flag = Counter(r["is_new_topic"] for r in all_records)
    print(f"\n📊  Old topics: {new_flag[False]}  |  New topics: {new_flag[True]}")
    print(f"\n{'Topic':<35} {'Count':>6}  {'Type'}")
    print(f"{'-'*35} {'-'*6}  {'-'*8}")
    for topic, count in sorted(counts.items()):
        is_new = topic in NEW_TOPIC_TEMPLATES
        tag = "🆕 NEW" if is_new else "  known"
        print(f"  {topic:<33} {count:>6}  {tag}")


if __name__ == "__main__":
    main()
