# ingestion.py

import os
import time
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# NEW: Import the unidecode library to handle special characters in IDs
from unidecode import unidecode

# --- Initialization ---
load_dotenv()

# --- Pinecone Setup ---
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")

# Check if the index exists and create it if it doesn't
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=768, # Dimension for intfloat/e5-base is 768
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait for the index to be ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    print("Index created successfully.")
else:
    print(f"Index '{index_name}' already exists.")

index = pc.Index(index_name)

# --- LangChain Setup ---
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# --- OPTIMIZED DOCUMENTS FOR SEMANTIC SEARCH ---
documents = [
    Document(
        page_content="Hank's Hefeweizen is a classic German-style wheat ale. It pours a hazy, deep golden color. The aroma and flavor are dominated by prominent notes of banana and a hint of spicy clove, all produced by the special Weihenstephan yeast. It has a soft, full-bodied mouthfeel.",
        metadata={"type": "beer", "name": "Hank's Hefeweizen", "abv": "4.6%", "ibu": 13, "style": "Hefeweizen"}
    ),
    Document(
        page_content="Kölsch Ale is a crisp, clean, and brilliantly clear ale. It offers a subtle malt sweetness and a light, delicate balance from the hops. It's an exceptionally smooth and easy-drinking beer, perfect for any occasion.",
        metadata={"type": "beer", "name": "Kölsch Ale", "abv": "4.7%", "ibu": 22, "style": "Kölsch"}
    ),
    Document(
        page_content="Pittsburgh Pilsner is a crisp and highly refreshing lager. It features a delicate balance of floral hop aroma and a clean, cracker-like malt backbone. Its light body and smooth, dry finish make it a perfect choice for warm days.",
        metadata={"type": "beer", "name": "Pittsburgh Pilsner", "abv": "5.6%", "ibu": 33, "style": "Pilsner"}
    ),
    Document(
        page_content="Knit Wit Ale is a Belgian-style wheat beer that pours hazy and pale. It delivers a vibrant blend of zesty citrus, primarily orange peel, and a subtle, spicy complexity from coriander. It's exceptionally crisp and refreshing with a soft, lingering finish.",
        metadata={"type": "beer", "name": "Knit Wit Ale", "abv": "5.4%", "ibu": 11, "style": "Witbier"}
    ),
    Document(
        page_content="Killer Lite is a classic American light lager. It is intentionally light in flavor and bitterness, making it incredibly easy to drink. Brewed with barley and rice flakes for a clean, crisp, and familiar taste.",
        metadata={"type": "beer", "name": "Killer Lite", "abv": "4.7%", "ibu": 8, "style": "Light Lager"}
    ),
    Document(
        page_content="Everyday IPA is a highly sessionable India Pale Ale, inspired by Founder's All Day IPA. It's brewed with local NY Craft 2 Row barley and oats, and hopped with Amarillo and Simcoe for bright citrus and pine notes. All the hop flavor without the high alcohol.",
        metadata={"type": "beer", "name": "Everyday IPA", "abv": "4.1%", "ibu": 30, "style": "IPA"}
    ),
    Document(
        page_content="Cider Creek: Cran Mango is a vibrant and fruity hard cider. It blends the tropical sweetness of mango with the tartness of cranberry for a perfectly balanced semi-sweet finish. Made from 100% New York State apples, this cider is naturally gluten-free.",
        metadata={"type": "cider", "name": "Cider Creek Cran Mango", "abv": "6.9%", "gluten_free": True, "style": "Fruit Cider"}
    ),
    Document(
        page_content="Bolivar Black Gold is a smooth dark ale. It features a rich complexity of roasted malt flavors, with hints of coffee and dark chocolate, leading to a satisfyingly smooth finish.",
        metadata={"type": "beer", "name": "Bolivar Black Gold", "abv": "5.8%", "ibu": 30, "style": "Dark Ale"}
    ),
    Document(
        page_content="Chocolate Peanut Butter Porter is a decadent and dessert-like dark beer. It's brewed with real chocolate and peanut butter, creating a harmonious blend of sweet, nutty, and lightly roasted flavors. A perfect treat for chilly days.",
        metadata={"type": "beer", "name": "Chocolate Peanut Butter Porter", "abv": "4.5%", "ibu": 23, "style": "Porter"}
    ),
    Document(
        page_content="Cream Ale is an exceptionally smooth and easy-drinking beer. It has a subtle malt sweetness, a characteristic hint of creaminess from corn, and a very mild hop presence to keep it balanced. A true American classic.",
        metadata={"type": "beer", "name": "Cream Ale", "abv": "5.1%", "ibu": 9, "style": "Cream Ale"}
    ),
    Document(
        page_content="Our Irish Red Ale is a malt-focused beer, brewed with pale malt and a touch of roasted barley. This gives it a beautiful ruby-red color and a smooth, slightly sweet taste with hints of caramel and toffee, finishing with a clean, dry character.",
        metadata={"type": "beer", "name": "Irish Red Ale", "abv": "4.7%", "ibu": 24, "style": "Red Ale"}
    ),
    Document(
        page_content="Wells Oatmeal Stout is a classic English-style stout. Brewed with oats to create a full body and a silky, smooth mouthfeel. It's dark and rich, with robust flavors of freshly roasted coffee, dark chocolate, and a hint of sweetness.",
        metadata={"type": "beer", "name": "Wells Oatmeal Stout", "abv": "4.5%", "ibu": 29, "style": "Stout"}
    ),
    Document(
        page_content="Sweet Summertime Kölsch Ale is a fruit-forward twist on a classic style. This bright golden ale is bursting with tropical notes of pineapple and mango, complemented by a honeyed malt sweetness and delicate floral hops. A truly crisp and refreshing summer beer.",
        metadata={"type": "beer", "name": "Sweet Summertime Kölsch Ale", "abv": "4.7%", "ibu": 22, "style": "Fruit Kölsch"}
    ),
    Document(
        page_content="Our Hard Seltzer is a crisp, refreshing, and lightly sparkling beverage. It serves as a perfect light alternative and is available in five distinct fruit flavors: Black Cherry, Mango, Wildberry, Coconut, and Citrus.",
        metadata={"type": "seltzer", "name": "Hard Seltzer", "abv": "4.7%", "flavors": ["Black Cherry", "Mango", "Wildberry", "Coconut", "Citrus"]}
    ),
    Document(
        page_content="The Wellsville Brewing Company offers a classy and comfortable tasting room, distinct from a typical bar scene. We serve light shareable plates, daily soup and sandwich specials, and provide tavern puzzles and board games for our guests to enjoy.",
        metadata={"type": "brewery_info", "category": "tasting_room", "name": "tasting-room-info"}
    ),
    Document(
        page_content="The brewery is located at 104 North Main Street, Wellsville, NY 14895. We foster a cozy, family-friendly atmosphere perfect for enjoying a drink and conversation. You can reach us at (585) 296-3230.",
        metadata={"type": "location", "address": "104 North Main Street, Wellsville, NY 14895", "phone": "(585) 296-3230", "name": "location-info"}
    ),
    Document(
        page_content="The Wellsville Brewing Company first opened its doors on October 6, 2018. It is housed in a beautifully renovated building on historic Main Street in the heart of Wellsville, New York.",
        metadata={"type": "brewery_info", "category": "history", "opened": "2018-10-06", "name": "history-info"}
    ),
    Document(
        page_content="As of March 2025, the Wellsville Brewing Company is under the proud new ownership of the Mariotti family: Kera, Joe, Isabel, and JT Mariotti.",
        metadata={"type": "ownership", "owners": "Mariotti family", "name": "ownership-info"}
    ),
    Document(
        page_content="Our operating hours are Thursday and Friday from 3:00 PM to 9:00 PM, and Saturday from 2:00 PM to 9:00 PM.",
        metadata={"type": "hours", "Thursday": "15:00-21:00", "Friday": "15:00-21:00", "Saturday": "14:00-21:00", "name": "hours-info"}
    )
]

# --- Data Ingestion ---

# Generate stable, compliant IDs from the document metadata
ids_raw = [doc.metadata.get("name", f"info-{i}") for i, doc in enumerate(documents)]

# Clean up IDs to be Pinecone-compliant (ASCII)
ids = [unidecode(str(id).replace(" ", "-").lower()) for id in ids_raw]

# Use upsert for adding/updating documents. This will overwrite existing vectors with the same ID.
print("Adding/updating documents in Pinecone vector store...")
vector_store.add_documents(documents=documents, ids=ids)
print(f"Successfully added/updated {len(documents)} documents.")