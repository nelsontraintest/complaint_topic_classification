"""
generate_dataset.py
───────────────────
Generates a large labeled complaint dataset (~2,000 records) for 12 banking topics.

Usage (PyCharm):  Run this file directly, or via terminal:
    python src/data_generation/generate_dataset.py

Output:
    data/complaints_labeled.csv
"""

import random
import csv
import os
from pathlib import Path

import yaml
from faker import Faker

fake = Faker()
random.seed(42)

# ─── Load Config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

OUTPUT_PATH = ROOT / CONFIG["data"]["labeled_dataset"]
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─── Complaint Templates per Topic ────────────────────────────────────────────
TEMPLATES = {
    "mobile_app_login": [
        "I cannot log in to your mobile banking app. It keeps saying 'invalid credentials' even though my password is correct.",
        "The banking app on my {device} won't let me sign in. I've tried resetting my password three times but the login still fails.",
        "Your mobile app login page just spins and never loads. I've been unable to access my account since {date}.",
        "Every time I try to log into the {bank} app, it shows an error code {code} and kicks me out.",
        "The fingerprint login on my banking app suddenly stopped working. I cannot get into my account at all.",
        "I updated the app last night and now I can't log in. The screen goes blank after I enter my credentials.",
        "Mobile app keeps logging me out immediately after I sign in. This is happening on my {device}.",
        "I'm getting 'Session expired' error right after I log in. Cannot stay logged into the app.",
        "The app asks for my password then immediately freezes. I have to force close it and start over.",
        "Login button is not responding on the mobile banking app. I've tapped it dozens of times.",
        "Cannot sign into mobile banking — it keeps asking me to re-register my device even though I've done it multiple times.",
        "App is stuck on the loading screen after I enter my login details. This has happened for two days straight.",
    ],
    "mobile_app_crash": [
        "Your mobile banking app crashes every time I try to open it on my {device}.",
        "The {bank} app freezes whenever I navigate to the transaction history page. I have to restart my phone.",
        "App closes unexpectedly while I'm in the middle of making a payment. Very frustrating.",
        "Mobile banking app is extremely slow. It takes over 2 minutes to load my account balance.",
        "The app crashes after the latest update {version}. Please fix this urgently.",
        "I'm experiencing constant app crashes on my {device}. The app is unusable right now.",
        "Banking app keeps crashing after I tap on 'View Statement'. Other features work fine.",
        "The app freezes for 5 minutes every time I try to view my credit card bill.",
        "Mobile app is unresponsive after I switch between accounts. I have to reinstall it every time.",
        "App crashes when I try to upload a photo of my check for mobile deposit.",
        "Your app has been crashing non-stop since the latest iOS update. Please release a patch.",
        "The banking app goes completely black and then crashes when I try to access my loan details.",
    ],
    "mobile_app_otp": [
        "I'm not receiving the OTP SMS needed to complete my transaction. I've been waiting 30 minutes.",
        "The one-time password your app sends has already expired by the time I receive it on my phone.",
        "OTP is not being sent to my registered mobile number {phone}. I cannot complete my login.",
        "I keep requesting OTP but it never arrives. I've checked my signal and SMS inbox.",
        "The 2FA code sent by your bank is always invalid even though I enter it right away.",
        "My OTP SMS is arriving 20 minutes late, causing the code to expire before I can use it.",
        "I've requested OTP 10 times in the last hour and haven't received a single one.",
        "The authentication code you sent is 6 digits but the app field only accepts 4 digits.",
        "OTP verification keeps failing. The code you send doesn't match what the app expects.",
        "I'm not getting 2FA codes via your banking app. My phone number is correct in your system.",
        "The mobile app is saying my OTP is incorrect even though I'm entering it exactly as received.",
        "SMS OTP is going to my old phone number even after I updated it in the app.",
    ],
    "mobile_app_transfer": [
        "I tried to transfer {amount} to my savings account via the mobile app but the transaction failed and the money is gone.",
        "Fund transfer on the mobile banking app shows 'Pending' for over 24 hours. The amount has been deducted.",
        "I attempted an urgent bank transfer but the app gave an error and I cannot see if it went through.",
        "Mobile app transfer failed midway. The money left my account but the recipient didn't receive it.",
        "I'm getting 'Transaction declined' when trying to transfer funds, but my balance is sufficient.",
        "The interbank transfer feature in your app has been broken for 3 days. I urgently need to move funds.",
        "App shows transfer was successful but my recipient's bank says no funds received.",
        "I cannot set a beneficiary in the mobile app to make a transfer. The 'Add Payee' button doesn't work.",
        "Transfer limit in the mobile app was changed without my consent. Now I cannot send large amounts.",
        "Mobile banking app doubled my transfer — debited my account twice for a single transaction.",
        "I'm unable to make overseas transfers via the app. It keeps timing out at the final step.",
        "The transfer confirmation page in the app keeps refreshing without completing the transaction.",
    ],
    "credit_card_declined": [
        "My credit card ending in {card_last4} was declined at a {merchant} store even though I have available credit.",
        "I'm embarrassed — my {bank} credit card was declined in front of everyone at the restaurant, even with sufficient credit limit.",
        "Credit card declined for an online purchase of {amount}. My card is active and not expired.",
        "My credit card was declined when I tried to pay for my flight tickets. This is urgent.",
        "I keep getting 'Card declined' error when using my credit card for contactless payments.",
        "My credit card transaction was declined at a supermarket. I had more than enough credit available.",
        "Credit card declined for a recurring subscription payment I've been making for 2 years.",
        "I couldn't pay my hotel bill because your credit card was declined. This caused serious embarrassment.",
        "My card was declined for a small transaction of {amount}. This makes no sense given my credit limit.",
        "Credit card is being declined for international online purchases. Please check if international transactions are blocked.",
        "I tried to use my credit card at {merchant} but it was declined without any reason.",
        "My credit card keeps getting declined even though I paid off the full balance last week.",
    ],
    "credit_card_billing": [
        "I see a charge of {amount} on my credit card statement that I did not authorize.",
        "I was billed twice for the same transaction at {merchant} on my credit card.",
        "My credit card statement shows a late fee even though I paid the minimum amount on time.",
        "There is an unknown charge from a merchant I've never heard of on my credit card bill.",
        "My credit card interest rate was changed without prior notification. My bill is much higher than expected.",
        "I returned a product but the refund hasn't appeared on my credit card statement after 3 weeks.",
        "I'm being charged an annual fee I never agreed to on my credit card account.",
        "My credit card bill shows a foreign transaction fee for a purchase I made domestically.",
        "I see charges on my credit card from a date when the card was in my wallet and unused.",
        "My credit card minimum payment due is incorrect. The calculation doesn't match my statement.",
        "I was charged {amount} for a service I cancelled two months ago. Still appearing on my card.",
        "Credit card reward points were deducted from my account without my knowledge.",
    ],
    "debit_card_blocked": [
        "My debit card was suddenly blocked. I cannot use it for any transactions.",
        "I tried to use my {bank} debit card and got a 'Card restricted' message. I haven't done anything wrong.",
        "My debit card was blocked after one failed PIN attempt. This is very inconvenient.",
        "I traveled abroad and my debit card was blocked without warning. I have no access to cash.",
        "Debit card shows as frozen in the app but I didn't freeze it. Please unblock it.",
        "My debit card was blocked due to 'suspicious activity' but all transactions were made by me.",
        "I cannot use my debit card online. It's being declined for all internet purchases.",
        "My debit card was blocked overnight. I need it for daily expenses and this is causing hardship.",
        "The bank blocked my debit card without sending me a notification or reason.",
        "My debit card works at ATMs but is declined at POS terminals in shops.",
        "Debit card suddenly stopped working. I called customer service but they couldn't explain why.",
        "My new debit card was blocked before I even activated it. I cannot make any purchases.",
    ],
    "debit_card_atm": [
        "ATM swallowed my debit card and didn't dispense the {amount} I requested.",
        "I tried to withdraw {amount} from the ATM but it failed and I was still charged.",
        "The ATM shows my transaction was successful but didn't give me the cash.",
        "ATM withdrew money from my account but dispensed the wrong amount.",
        "I inserted my debit card in your ATM and it got stuck. The machine ate my card.",
        "ATM gave me an error message and kept my card. Transaction reference: {ref}.",
        "I was charged for an ATM withdrawal that never happened. The machine malfunctioned.",
        "The ATM near {location} is broken and captured my debit card. Please retrieve it urgently.",
        "Debit card declined at ATM even though I entered the correct PIN and have sufficient balance.",
        "ATM dispensed short of the requested amount but debited my account for the full amount.",
        "My debit card got stuck in the ATM slot. It won't come out and the screen has gone blank.",
        "I did a balance inquiry at the ATM and it showed a lower balance than what's in my account.",
    ],
    "etc_card_toll": [
        "My ETC card was not recognized at the toll booth on the highway today.",
        "I drove through the toll lane and the ETC system didn't detect my card, causing a fine.",
        "ETC card reader at the {location} toll failed to scan my card. The barrier didn't open.",
        "My ETC card always worked before but suddenly stopped being recognized at toll gates.",
        "I received a toll violation notice even though my ETC card was properly installed.",
        "The toll system said my ETC card was invalid even though it's registered and topped up.",
        "My ETC card is not being detected at any toll booth since yesterday.",
        "ETC card transponder seems to have failed. I'm getting error beeps at every toll plaza.",
        "I was fined for not paying toll because my ETC card wasn't recognized. Please help.",
        "The toll plaza scanner rejected my ETC card three times. I had to pay cash as a result.",
        "ETC card works at some toll booths but not others. Inconsistent behavior is causing issues.",
        "My ETC card registration shows active but the toll system keeps rejecting it.",
    ],
    "etc_card_topup": [
        "I tried to top up my ETC card via the mobile app but the top-up failed and money was deducted.",
        "My ETC card top-up hasn't been credited even after {hours} hours. Transaction reference: {ref}.",
        "I can't find the ETC card top-up option in the banking app anymore. It seems to have disappeared.",
        "Online top-up for my ETC card failed but the payment was debited from my bank account.",
        "ETC card auto top-up feature stopped working. My card balance is zero and I can't pay tolls.",
        "I topped up my ETC card {amount} but the toll system still shows zero balance.",
        "ETC card top-up via ATM was successful but the balance on the card wasn't updated.",
        "I've been waiting 3 days for an ETC top-up to reflect on my card. Customer service is not helping.",
        "The top-up limit for my ETC card was reduced without notice. I cannot add the amount I need.",
        "ETC card auto reload failed multiple times this month, leaving me unable to use the highway.",
        "Top-up transaction for my ETC card is showing 'processing' for 48 hours with no update.",
        "I tried to reload my ETC card at a top-up kiosk and it rejected my debit card.",
    ],
    "account_balance": [
        "My account balance is showing incorrect information. There is {amount} missing that I cannot account for.",
        "My savings account balance dropped by {amount} overnight with no transaction showing on my statement.",
        "I transferred money in but my account balance hasn't updated. The funds seem to have disappeared.",
        "My account statement shows a different balance than what the ATM and app display.",
        "There are duplicate debits on my account that don't match any transaction I made.",
        "My account balance shows negative even though I have funds deposited. This looks like a system error.",
        "I deposited a cheque 5 days ago but it hasn't appeared in my account balance.",
        "My salary was credited but an unauthorized debit brought my balance to nearly zero.",
        "My fixed deposit maturity amount was not credited to my account on the due date.",
        "I've noticed regular small debits from my account that I don't recognize. Possible fraud.",
        "My account shows a credit reversal that I never initiated. Balance is now lower than expected.",
        "I can see my pending transactions but they never cleared, and my available balance is wrong.",
    ],
    "online_banking_access": [
        "I cannot access your internet banking portal. The website keeps showing a '503 Service Unavailable' error.",
        "Online banking login page is down. I've tried multiple browsers and devices.",
        "Your internet banking website is extremely slow and times out before I can complete any transaction.",
        "I'm locked out of online banking after too many incorrect password attempts. Please reset my access.",
        "The online banking portal redirects me to an error page every time I try to log in.",
        "Internet banking is not loading on my computer. I need to transfer funds urgently.",
        "Your bank's website certificate has expired and my browser won't let me access online banking.",
        "Online banking dashboard shows all zeros — my accounts and transactions are not displaying.",
        "I successfully log into online banking but all menu options are greyed out and unclickable.",
        "Online banking session times out in 30 seconds even when I'm actively using it.",
        "I'm unable to reset my online banking password. The 'Forgot Password' link doesn't work.",
        "Online banking shows my old address and I cannot update my profile. The save button is broken.",
    ],
}

BANKS = ["MegaBank", "FirstCity Bank", "TrustBank", "PrimeBank", "NexBank", "Unified Bank", "CitiCore Bank"]
DEVICES = ["iPhone 15", "Samsung Galaxy S24", "iPhone 14 Pro", "Pixel 8", "Huawei P60", "OnePlus 12"]
MERCHANTS = ["McDonald's", "Grab", "Shopee", "Lazada", "IKEA", "Starbucks", "Shell", "Watsons", "AEON Mall"]
LOCATIONS = ["Bangsar", "KLCC", "Petaling Jaya", "Subang", "Cheras", "Mont Kiara", "Damansara"]


def fill_template(template: str) -> str:
    """Replace placeholders in a template with realistic fake values."""
    return (
        template
        .replace("{device}", random.choice(DEVICES))
        .replace("{bank}", random.choice(BANKS))
        .replace("{date}", fake.date_this_year().strftime("%B %d, %Y"))
        .replace("{code}", str(random.randint(1000, 9999)))
        .replace("{amount}", f"RM {random.randint(50, 5000):,}.{random.randint(0, 99):02d}")
        .replace("{phone}", fake.phone_number())
        .replace("{card_last4}", str(random.randint(1000, 9999)))
        .replace("{merchant}", random.choice(MERCHANTS))
        .replace("{ref}", fake.bothify(text="TXN-????-######").upper())
        .replace("{hours}", str(random.randint(12, 72)))
        .replace("{location}", random.choice(LOCATIONS))
        .replace("{version}", f"v{random.randint(5, 9)}.{random.randint(0, 9)}.{random.randint(0, 9)}")
        .replace("{version}", f"v{random.randint(5, 9)}.{random.randint(0, 9)}.{random.randint(0, 9)}")
    )


def add_variation(text: str) -> str:
    """Add natural language variations: prepend, append, or paraphrase context."""
    prefixes = [
        "Hi, I'm writing to complain that ",
        "I am extremely frustrated because ",
        "I need urgent help — ",
        "This is unacceptable. ",
        "I've been a loyal customer for years and now ",
        "Dear support team, ",
        "I'm contacting you because ",
        "This is my third complaint about this issue. ",
        "",  # No prefix — keep original
        "",
        "",
    ]
    suffixes = [
        " Please resolve this immediately.",
        " This is affecting my daily life.",
        " I expect a response within 24 hours.",
        " I will escalate to the regulator if this is not resolved.",
        " I'm considering switching banks over this.",
        "",
        "",
        "",
    ]
    prefix = random.choice(prefixes)
    suffix = random.choice(suffixes)
    if prefix:
        # Lowercase first letter of original text when adding prefix
        text = prefix + text[0].lower() + text[1:]
    return text + suffix


def generate_records(target_per_topic: int = 170) -> list[dict]:
    """Generate complaint records for all known topics."""
    records = []
    for topic, templates in TEMPLATES.items():
        count = 0
        while count < target_per_topic:
            template = random.choice(templates)
            text = fill_template(template)
            text = add_variation(text)
            records.append({"complaint_text": text, "topic": topic})
            count += 1
    random.shuffle(records)
    return records


def main():
    print("🔄  Generating labeled complaint dataset...")
    records = generate_records(target_per_topic=170)  # 12 topics × 170 = 2,040 records
    print(f"✅  Generated {len(records):,} records across {len(TEMPLATES)} topics")

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["complaint_text", "topic"])
        writer.writeheader()
        writer.writerows(records)

    print(f"💾  Saved to: {OUTPUT_PATH}")

    # ── Print distribution ─────────────────────────────────────────────────────
    from collections import Counter
    counts = Counter(r["topic"] for r in records)
    print("\n📊  Topic Distribution:")
    print(f"  {'Topic':<35} {'Count':>6}")
    print(f"  {'-'*35} {'-'*6}")
    for topic, count in sorted(counts.items()):
        print(f"  {topic:<35} {count:>6}")


if __name__ == "__main__":
    main()
