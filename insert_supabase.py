import os
from dotenv import load_dotenv
from supabase import create_client, Client

# --- Data to be Inserted ---
documents = [
    {
        "page_content": "Hank's Hefeweizen is a hazy, deep golden ale brewed with Weihenstephan yeast that produces banana and clove notes.",
        "metadata": {"type": "beer", "name": "Hank's Hefeweizen", "abv": "4.6%", "ibu": 13, "style": "Hefeweizen"}
    },
    {
        "page_content": "Kölsch Ale is a crisp and clean ale with subtle malt sweetness and light hop balance.",
        "metadata": {"type": "beer", "name": "Kölsch Ale", "abv": "4.7%", "ibu": 22, "style": "Kölsch"}
    },
    {
        "page_content": "Pittsburgh Pilsner is crisp and refreshing, with a delicate balance of floral hops and a clean malt backbone. Light body and smooth finish make it perfect for warm days.",
        "metadata": {"type": "beer", "name": "Pittsburgh Pilsner", "abv": "5.6%", "ibu": 33, "style": "Pilsner"}
    },
    {
        "page_content": "Knit Wit Ale delivers a vibrant blend of citrus and subtle spice in a smooth, hazy wheat beer. Crisp and refreshing with a soft, lingering finish.",
        "metadata": {"type": "beer", "name": "Knit Wit Ale", "abv": "5.4%", "ibu": 11, "style": "Witbier"}
    },
    {
        "page_content": "Killer Lite is similar to the mass-produced American brands: light in flavor, light in bitterness, brewed with water, barley, rice flakes and yeast.",
        "metadata": {"type": "beer", "name": "Killer Lite", "abv": "4.7%", "ibu": 8, "style": "Light Lager"}
    },
    {
        "page_content": "Everyday IPA is a clone of Founder's All Day IPA. Brewed with local NY Craft 2 Row barley and oats, hopped with Amarillo and Simcoe. Sessionable at 4.1% ABV.",
        "metadata": {"type": "beer", "name": "Everyday IPA", "abv": "4.1%", "ibu": 30, "style": "IPA"}
    },
    {
        "page_content": "Cider Creek: Cran Mango is a tropical and fruity cider made with 100% New York State apples, cranberry, and mango. Semi-sweet with low tartness. Gluten-free.",
        "metadata": {"type": "cider", "name": "Cider Creek Cran Mango", "abv": "6.9%", "gluten_free": True, "style": "Fruit Cider"}
    },
    {
        "page_content": "Bolivar Black Gold is a dark ale with roasted malt complexity and smooth finish.",
        "metadata": {"type": "beer", "name": "Bolivar Black Gold", "abv": "5.8%", "ibu": 30, "style": "Dark Ale"}
    },
    {
        "page_content": "Chocolate Peanut Butter Porter is brewed with chocolate and peanut butter. Slightly sweet, lightly roasted — perfect for chilly winter days.",
        "metadata": {"type": "beer", "name": "Chocolate Peanut Butter Porter", "abv": "4.5%", "ibu": 23, "style": "Porter"}
    },
    {
        "page_content": "Cream Ale is smooth and easy-drinking with subtle malt sweetness, a hint of creaminess, and mild hop presence.",
        "metadata": {"type": "beer", "name": "Cream Ale", "abv": "5.1%", "ibu": 9, "style": "Cream Ale"}
    },
    {
        "page_content": "Irish Red Ale is a malt-forward ale brewed with pale malt and roasted barley, giving it a beautiful red color and smooth taste.",
        "metadata": {"type": "beer", "name": "Irish Red Ale", "abv": "4.7%", "ibu": 24, "style": "Red Ale"}
    },
    {
        "page_content": "Wells Oatmeal Stout is based on an English stout. Brewed with oats for body and silkiness, it's dark and rich with flavors of coffee and chocolate.",
        "metadata": {"type": "beer", "name": "Wells Oatmeal Stout", "abv": "4.5%", "ibu": 29, "style": "Stout"}
    },
    {
        "page_content": "Sweet Summertime Kölsch Ale is bright and golden, with honeyed malt and floral hops. Bursting with pineapple and mango, crisp and refreshing.",
        "metadata": {"type": "beer", "name": "Sweet Summertime Kölsch Ale", "abv": "4.7%", "ibu": 22, "style": "Fruit Kölsch"}
    },
    {
        "page_content": "Hard Seltzer is crisp, refreshing, and lightly sparkling. Offered in five flavors: Black Cherry, Mango, Wildberry, Coconut, and Citrus.",
        "metadata": {"type": "seltzer", "name": "Hard Seltzer", "abv": "4.7%"}
    }
]

def populate_drinks_table():
    load_dotenv()

    # --- Connect to Supabase ---
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_API_KEY")

    if not url or not key:
        raise ValueError("Supabase URL or Key not found in environment variables.")

    supabase: Client = create_client(url, key)

    # --- Prepare Data for Insertion ---
    drinks_to_insert = []
    for doc in documents:
        meta = doc["metadata"]
        
        # Skip non-drink items
        if meta.get("type") not in ["beer", "cider", "seltzer"]:
            continue

        # Clean up ABV data: remove '%' and convert to a number
        abv_str = meta.get("abv", "0%").replace('%', '')
        try:
            abv_float = float(abv_str)
        except ValueError:
            abv_float = None # Handle cases where conversion might fail

        drink_record = {
            "name": meta.get("name"),
            "type": meta.get("type"),
            "style": meta.get("style"),
            "abv": abv_float,
            "ibu": meta.get("ibu"),
            "gluten_free": meta.get("gluten_free", False),
            "description": doc["page_content"]
        }
        drinks_to_insert.append(drink_record)

    # --- Insert Data into Supabase ---
    print(f"Attempting to insert {len(drinks_to_insert)} records into the 'drinks' table...")
    try:
        # The 'insert' method takes a list of dictionaries
        response = supabase.table("drinks").insert(drinks_to_insert).execute()
        print("Successfully inserted data!")
        print("Response:", response.data)
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Run the function ---
if __name__ == "__main__":
    populate_drinks_table()