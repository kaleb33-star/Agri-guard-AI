"""
AgriGuard Ethiopia - SMS / USSD Demonstration

This is a local demonstration of the future USSD/SMS farmer interface.
It uses the shared AgriGuard intelligence engine.

The farmer provides:
    - Region
    - Crop
    - Cultivated area
    - Year

The engine automatically obtains:
    - Sentinel-2 NDVI through Google Earth Engine
    - Historical temperature through Open-Meteo
    - Historical rainfall through Open-Meteo

Then all three trained models are run.
"""

import agriguard_engine


REGIONS = [
    "Afar",
    "Amhara",
    "Benushangul Gumuz",
    "DIRE DAWA",
    "Gambela",
    "Harari",
    "Oromiya",
    "S.N.N.P.R",
    "Somalia",
    "Tigray",
]

CROPS = [
    "Teff",
    "Barely",
    "Wheat",
    "Maize",
    "Millet",
    "Sorghum",
    "Oats",
]


def choose_from_list(title, options):
    print()
    print(title)

    for number, option in enumerate(options, start=1):
        print(f"{number}. {option}")

    while True:
        try:
            choice = int(input("Choose a number: "))

            if 1 <= choice <= len(options):
                return options[choice - 1]

            print("Please choose one of the numbers shown.")

        except ValueError:
            print("Please enter a number.")


def main():
    print()
    print("=" * 60)
    print("AGUARD ETHIOPIA - USSD/SMS DEMO")
    print("=" * 60)

    print()
    print("Example farmer session")
    print("----------------------")

    region = choose_from_list(
        "Select your region:",
        REGIONS,
    )

    crop = choose_from_list(
        "Select your crop:",
        CROPS,
    )

    while True:
        try:
            area = float(
                input("Cultivated area in hectares: ")
            )

            if area <= 0:
                print("Area must be greater than zero.")
                continue

            break

        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            year = int(
                input("Year (2015 or later): ")
            )

            if year < 2015:
                print("The current real-data pipeline starts at 2015.")
                continue

            break

        except ValueError:
            print("Please enter a valid year.")

    print()
    print("AgriGuard is collecting satellite and weather information.")
    print("Please wait...")

    try:
        # IMPORTANT:
        # analyze_farm() expects crop_type and area_cultivated_ha.
        # Do NOT use crop= or area_ha= here.
        result = agriguard_engine.analyze_farm(
            region=region,
            crop_type=crop,
            area_cultivated_ha=area,
            year=year,
        )

    except Exception as error:
        print()
        print("❌ Analysis failed.")
        print(error)
        return

    yield_data = result.get(
        "yield_prediction",
        {}
    )

    performance = result.get(
        "crop_performance",
        {}
    )

    risk = result.get(
        "yield_risk",
        {}
    )

    print()
    print("=" * 60)
    print("AGRIGUARD RESULT")
    print("=" * 60)

    print()
    print(f"Region: {region}")
    print(f"Crop: {crop}")
    print(f"Area: {area:,.2f} ha")
    print(f"Year: {year}")

    print()
    print(
        "Expected Yield: "
        f"{yield_data.get('yield_kg_ha', 'N/A'):,.2f} kg/ha"
        if isinstance(yield_data.get("yield_kg_ha"), (int, float))
        else
        f"Expected Yield: {yield_data.get('yield_kg_ha', 'N/A')}"
    )

    production = yield_data.get("production_kg")

    if isinstance(production, (int, float)):
        print(
            f"Expected Production: {production:,.2f} kg"
        )
    else:
        print(
            f"Expected Production: {production or 'N/A'}"
        )

    print()
    print(
        "Performance: "
        f"{performance.get('class_name', 'N/A')}"
    )

    confidence = performance.get("confidence")

    if isinstance(confidence, (int, float)):
        print(
            f"Performance Confidence: "
            f"{confidence * 100:.1f}%"
        )

    print(
        "Yield Risk: "
        f"{risk.get('class_name', 'N/A')}"
    )

    confidence = risk.get("confidence")

    if isinstance(confidence, (int, float)):
        print(
            f"Risk Confidence: "
            f"{confidence * 100:.1f}%"
        )

    environmental = result.get(
        "environmental_features",
        {}
    )

    print()
    print("Environmental Intelligence")
    print("---------------------------")

    print(
        f"NDVI Mean: "
        f"{environmental.get('NDVI_mean', 'N/A')}"
    )

    print(
        f"NDVI Peak: "
        f"{environmental.get('NDVI_peak', 'N/A')}"
    )

    print(
        f"Mean Temperature: "
        f"{environmental.get('temperature_mean', 'N/A')} °C"
    )

    print(
        f"Rainfall: "
        f"{environmental.get('rainfall', 'N/A')} mm"
    )

    print()
    print("📱 SMS-style result")
    print("-------------------")

    try:
        sms_message = agriguard_engine.create_sms_summary(
            result
        )
        print(sms_message)

    except Exception as error:
        print(
            "SMS summary could not be generated:"
        )
        print(error)

    print()
    print("✅ Analysis completed.")


if __name__ == "__main__":
    main()
