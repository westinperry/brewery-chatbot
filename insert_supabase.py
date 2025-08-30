import os
from dotenv import load_dotenv
from supabase import create_client, Client

# --- Data to be Inserted ---
documents = [
    {
        "page_content": "Hank's Hefeweizen is a hazy, deep golden ale brewed with Weihenstephan yeast that produces banana and clove notes.",
        "metadata": {
            "type": "beer", "name": "Hank's Hefeweizen", "abv": "4.6%", "ibu": 13, "style": "Hefeweizen",
            "is_hazy": True, "flavor_notes": ["banana", "clove"], "key_ingredients": ["Weihenstephan yeast"]
        }
    },
    {
        "page_content": "Kölsch Ale is a crisp and clean ale with subtle malt sweetness and light hop balance.",
        "metadata": {
            "type": "beer", "name": "Kölsch Ale", "abv": "4.7%", "ibu": 22, "style": "Kölsch",
            "is_hazy": False, "flavor_notes": ["malt sweetness", "clean"], "key_ingredients": []
        }
    },
    {
        "page_content": "Pittsburgh Pilsner is crisp and refreshing, with a delicate balance of floral hops and a clean malt backbone. Light body and smooth finish make it perfect for warm days.",
        "metadata": {
            "type": "beer", "name": "Pittsburgh Pilsner", "abv": "5.6%", "ibu": 33, "style": "Pilsner",
            "is_hazy": False, "flavor_notes": ["floral hops", "clean malt", "crisp"], "key_ingredients": []
        }
    },
    {
        "page_content": "Knit Wit Ale delivers a vibrant blend of citrus and subtle spice in a smooth, hazy wheat beer. Crisp and refreshing with a soft, lingering finish.",
        "metadata": {
            "type": "beer", "name": "Knit Wit Ale", "abv": "5.4%", "ibu": 11, "style": "Witbier",
            "is_hazy": True, "flavor_notes": ["citrus", "spice"], "key_ingredients": ["wheat"]
        }
    },
    {
        "page_content": "Killer Lite is similar to the mass-produced American brands: light in flavor, light in bitterness, brewed with water, barley, rice flakes and yeast.",
        "metadata": {
            "type": "beer", "name": "Killer Lite", "abv": "4.7%", "ibu": 8, "style": "Light Lager",
            "is_hazy": False, "flavor_notes": ["light"], "key_ingredients": ["barley", "rice flakes"]
        }
    },
    {
        "page_content": "Everyday IPA is a clone of Founder's All Day IPA. Brewed with local NY Craft 2 Row barley and oats, hopped with Amarillo and Simcoe. Sessionable at 4.1% ABV.",
        "metadata": {
            "type": "beer", "name": "Everyday IPA", "abv": "4.1%", "ibu": 30, "style": "IPA",
            "is_hazy": False, "flavor_notes": [], "key_ingredients": ["barley", "oats", "Amarillo hops", "Simcoe hops"]
        }
    },
    {
        "page_content": "Cider Creek: Cran Mango is a tropical and fruity cider made with 100% New York State apples, cranberry, and mango. Semi-sweet with low tartness. Gluten-free.",
        "metadata": {
            "type": "cider", "name": "Cider Creek Cran Mango", "abv": "6.9%", "gluten_free": True, "style": "Fruit Cider",
            "flavor_notes": ["tropical", "fruity", "cranberry", "mango", "semi-sweet"], "key_ingredients": ["NY apples", "cranberry", "mango"]
        }
    },
    {
        "page_content": "Bolivar Black Gold is a dark ale with roasted malt complexity and smooth finish.",
        "metadata": {
            "type": "beer", "name": "Bolivar Black Gold", "abv": "5.8%", "ibu": 30, "style": "Dark Ale",
            "is_hazy": False, "flavor_notes": ["roasted malt"], "key_ingredients": []
        }
    },
    {
        "page_content": "Chocolate Peanut Butter Porter is brewed with chocolate and peanut butter. Slightly sweet, lightly roasted — perfect for chilly winter days.",
        "metadata": {
            "type": "beer", "name": "Chocolate Peanut Butter Porter", "abv": "4.5%", "ibu": 23, "style": "Porter",
            "is_hazy": False, "flavor_notes": ["chocolate", "peanut butter", "sweet", "roasted"], "key_ingredients": ["chocolate", "peanut butter"]
        }
    },
    {
        "page_content": "Cream Ale is smooth and easy-drinking with subtle malt sweetness, a hint of creaminess, and mild hop presence.",
        "metadata": {
            "type": "beer", "name": "Cream Ale", "abv": "5.1%", "ibu": 9, "style": "Cream Ale",
            "is_hazy": False, "flavor_notes": ["smooth", "malt sweetness", "creamy"], "key_ingredients": []
        }
    },
    {
        "page_content": "Irish Red Ale is a malt-forward ale brewed with pale malt and roasted barley, giving it a beautiful red color and smooth taste.",
        "metadata": {
            "type": "beer", "name": "Irish Red Ale", "abv": "4.7%", "ibu": 24, "style": "Red Ale",
            "is_hazy": False, "flavor_notes": ["malt-forward", "smooth"], "key_ingredients": ["pale malt", "roasted barley"]
        }
    },
    {
        "page_content": "Wells Oatmeal Stout is based on an English stout. Brewed with oats for body and silkiness, it's dark and rich with flavors of coffee and chocolate.",
        "metadata": {
            "type": "beer", "name": "Wells Oatmeal Stout", "abv": "4.5%", "ibu": 29, "style": "Stout",
            "is_hazy": False, "flavor_notes": ["dark", "rich", "coffee", "chocolate"], "key_ingredients": ["oats"]
        }
    },
    {
        "page_content": "Sweet Summertime Kölsch Ale is bright and golden, with honeyed malt and floral hops. Bursting with pineapple and mango, crisp and refreshing.",
        "metadata": {
            "type": "beer", "name": "Sweet Summertime Kölsch Ale", "abv": "4.7%", "ibu": 22, "style": "Fruit Kölsch",
            "is_hazy": False, "flavor_notes": ["honeyed malt", "floral hops", "pineapple", "mango", "crisp"], "key_ingredients": ["pineapple", "mango"]
        }
    },
    {
        "page_content": "Hard Seltzer is crisp, refreshing, and lightly sparkling. Offered in five flavors: Black Cherry, Mango, Wildberry, Coconut, and Citrus.",
        "metadata": {
            "type": "seltzer", "name": "Hard Seltzer", "abv": "4.7%",
            "flavor_notes": ["Black Cherry", "Mango", "Wildberry", "Coconut", "Citrus"]
        }
    }
]


def populate_drinks_table():
    load_dotenv()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_API_KEY")

    if not url or not key:
        raise ValueError("Supabase URL or Key not found in environment variables.")

    supabase: Client = create_client(url, key)

    drinks_to_insert = []
    for doc in documents:
        meta = doc["metadata"]
        
        if meta.get("type") not in ["beer", "cider", "seltzer"]:
            continue

        abv_str = meta.get("abv", "0%").replace('%', '')
        try:
            abv_float = float(abv_str)
        except ValueError:
            abv_float = None

        drink_record = {
            "name": meta.get("name"),
            "type": meta.get("type"),
            "style": meta.get("style"),
            "abv": abv_float,
            "ibu": meta.get("ibu"),
            "gluten_free": meta.get("gluten_free", False),
            "description": doc["page_content"],
            # --- NEW FIELDS ---
            "is_hazy": meta.get("is_hazy", False),
            "flavor_notes": meta.get("flavor_notes", []),
            "key_ingredients": meta.get("key_ingredients", [])
        }
        drinks_to_insert.append(drink_record)

    print(f"Attempting to insert {len(drinks_to_insert)} records into the 'drinks' table...")
    try:
        response = supabase.table("drinks").insert(drinks_to_insert).execute()
        print("Successfully inserted data!")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- NEW FUNCTION TO LIST BEER STYLES ---
def list_beer_styles():
    """
    Connects to Supabase, queries for all unique beer styles, and prints them.
    """
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_API_KEY")

    if not url or not key:
        raise ValueError("Supabase URL or Key not found in environment variables.")

    supabase: Client = create_client(url, key)

    print("--- Querying for unique beer styles ---")
    try:
        response = supabase.table("drinks").select("style").eq("type", "beer").execute()
        
        if response.data:
            # Extract styles, remove None values, get unique values, and sort
            styles = [item['style'] for item in response.data if item.get('style')]
            unique_styles = sorted(list(set(styles)))
            
            print("Found the following unique beer styles:")
            for style in unique_styles:
                print(f"- {style}")
        else:
            print("No beer styles found.")

    except Exception as e:
        print(f"An error occurred while querying for styles: {e}")


# --- Run the functions ---
if __name__ == "__main__":
    # First, populate the table with all drink data
    populate_drinks_table()

    # Add a separator for cleaner output
    print("\n" + "="*30 + "\n")

    # Then, run the new function to list only the beer styles
    list_beer_styles()