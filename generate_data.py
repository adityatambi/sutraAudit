import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

def generate_applicant_data(num_applicants=1000):
    """
    Generates a synthetic dataset of 1,000 Indian credit applicants.
    Creates a 'baseline' group (formal signals) and a 'target' group (informal signals)
    where BOTH groups have mathematically identical ground-truth default rates (15%).
    """
    print(f"Generating {num_applicants} synthetic applicant profiles...")
    
    # 50% baseline (formal), 50% target (informal/marginalized)
    groups = np.random.choice(["baseline", "target"], size=num_applicants, p=[0.50, 0.50])
    
    # Financial indicators (identical distributions to enforce identical ground-truth default rates)
    income = np.random.normal(loc=45000, scale=8000, size=num_applicants)
    income = np.clip(income, 15000, 90000)
    
    debt_ratio = np.random.uniform(0.15, 0.55, size=num_applicants)
    
    # Ground-truth default rate: exactly 15.0% default rate for both groups
    # Generates a binary label where 0 = Repaid (Good), 1 = Default (Bad)
    ground_truth_default = np.zeros(num_applicants, dtype=int)
    for g in ["baseline", "target"]:
        indices = np.where(groups == g)[0]
        n_defaults = int(round(len(indices) * 0.15))
        default_indices = np.random.choice(indices, size=n_defaults, replace=False)
        ground_truth_default[default_indices] = 1

    
    # Generate Alternative Digital Footprint signals (Proxies) based on group division
    upi_velocity_10_50 = np.zeros(num_applicants, dtype=int)
    device_tier = np.zeros(num_applicants, dtype=int) # 0 = Premium, 1 = Lower-tier Android
    pincode_tier = np.zeros(num_applicants, dtype=int) # 0 = Urban, 1 = Rural
    regional_dialect_count = np.zeros(num_applicants, dtype=int)
    verifiable_income = np.zeros(num_applicants, dtype=int) # 0 = Self-declared, 1 = Salary slip
    
    for i in range(num_applicants):
        grp = groups[i]
        
        if grp == "baseline":
            # Formal signals
            upi_velocity_10_50[i] = np.random.poisson(lam=3) # Low velocity of small transfers
            device_tier[i] = np.random.choice([0, 1], p=[0.90, 0.10]) # Premium devices
            pincode_tier[i] = np.random.choice([0, 1], p=[0.95, 0.05]) # Urban areas
            regional_dialect_count[i] = np.random.poisson(lam=6) # Low regional transliterated messages
            verifiable_income[i] = np.random.choice([1, 0], p=[0.92, 0.08]) # High rate of salary slips
        else:
            # Informal / Marginalized signals
            upi_velocity_10_50[i] = np.random.poisson(lam=38) # High velocity of small ₹10-₹50 UPI transfers
            device_tier[i] = np.random.choice([0, 1], p=[0.08, 0.92]) # Lower-tier Android device metadata
            pincode_tier[i] = np.random.choice([0, 1], p=[0.15, 0.85]) # Rural pincodes
            regional_dialect_count[i] = np.random.poisson(lam=68) # High regional dialect/transliterated messages
            verifiable_income[i] = np.random.choice([1, 0], p=[0.10, 0.90]) # Self-declared income
            
    # Generate biased historical default labels
    # Simulates systemic prejudice in historical training data:
    # Favoring baseline group, over-reporting defaults for target group
    historical_default = np.zeros(num_applicants, dtype=int)
    for i in range(num_applicants):
        gtd = ground_truth_default[i]
        grp = groups[i]
        
        if gtd == 1:
            # True defaulters: highly likely to be flagged historically
            p_default = 0.95 if grp == "target" else 0.80
        else:
            # Non-defaulters: target group is erroneously flagged as default far more often
            p_default = 0.28 if grp == "target" else 0.06
            
        historical_default[i] = np.random.choice([1, 0], p=[p_default, 1 - p_default])
        
    # Create DataFrame
    df = pd.DataFrame({
        "applicant_id": [f"APP_{1000+i}" for i in range(num_applicants)],
        "group": groups,
        "upi_velocity_10_50": upi_velocity_10_50,
        "device_tier": device_tier,
        "pincode_tier": pincode_tier,
        "regional_dialect_count": regional_dialect_count,
        "verifiable_income": verifiable_income,
        "income": income.round(2),
        "debt_ratio": debt_ratio.round(4),
        "ground_truth_default": ground_truth_default,
        "historical_default": historical_default
    })
    
    return df

def main():
    df = generate_applicant_data(1000)
    
    # Save to CSV
    output_file = "applicants.csv"
    df.to_csv(output_file, index=False)
    print(f"Dataset successfully saved to '{output_file}'.")
    
    # Print some statistics to verify the setup
    print("\nDataset Verification Summary:")
    print("-" * 40)
    print(f"Total applicants: {len(df)}")
    print(f"Baseline group count: {sum(df['group'] == 'baseline')}")
    print(f"Target group count: {sum(df['group'] == 'target')}")
    
    print("\nGround-Truth Default Rates (should be identical):")
    print(df.groupby("group")["ground_truth_default"].mean().apply(lambda x: f"{x*100:.2f}%"))
    
    print("\nBiased Historical Default Rates (should be unequal due to bias):")
    print(df.groupby("group")["historical_default"].mean().apply(lambda x: f"{x*100:.2f}%"))
    
    print("\nProxy Feature Means by Group:")
    features = ["upi_velocity_10_50", "device_tier", "pincode_tier", "regional_dialect_count", "verifiable_income"]
    print(df.groupby("group")[features].mean())

if __name__ == "__main__":
    main()
