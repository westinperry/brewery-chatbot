# import basics
import os
import time
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import supabase
from supabase import create_client, Client

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE-API-KEY"))

index_name = os.environ.get("PINECONE-INDEX-NAME")

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base")

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

documents = [
    Document(
        page_content="Hank's Hefeweizen is a hazy, deep golden ale brewed with Weihenstephan yeast that produces banana and clove notes.",
        metadata={"type": "beer", "name": "Hank's Hefeweizen", "abv": "4.6%", "ibu": 13, "style": "Hefeweizen"}
    ),
    Document(
        page_content="Kölsch Ale is a crisp and clean ale with subtle malt sweetness and light hop balance.",
        metadata={"type": "beer", "name": "Kölsch Ale", "abv": "4.7%", "ibu": 22, "style": "Kölsch"}
    ),
    Document(
        page_content="Pittsburgh Pilsner is crisp and refreshing, with a delicate balance of floral hops and a clean malt backbone. Light body and smooth finish make it perfect for warm days.",
        metadata={"type": "beer", "name": "Pittsburgh Pilsner", "abv": "5.6%", "ibu": 33, "style": "Pilsner"}
    ),
    Document(
        page_content="Knit Wit Ale delivers a vibrant blend of citrus and subtle spice in a smooth, hazy wheat beer. Crisp and refreshing with a soft, lingering finish.",
        metadata={"type": "beer", "name": "Knit Wit Ale", "abv": "5.4%", "ibu": 11, "style": "Witbier"}
    ),
    Document(
        page_content="Killer Lite is similar to the mass-produced American brands: light in flavor, light in bitterness, brewed with water, barley, rice flakes and yeast.",
        metadata={"type": "beer", "name": "Killer Lite", "abv": "4.7%", "ibu": 8, "style": "Light Lager"}
    ),
    Document(
        page_content="Everyday IPA is a clone of Founder's All Day IPA. Brewed with local NY Craft 2 Row barley and oats, hopped with Amarillo and Simcoe. Sessionable at 4.1% ABV.",
        metadata={"type": "beer", "name": "Everyday IPA", "abv": "4.1%", "ibu": 30, "style": "IPA"}
    ),
    Document(
        page_content="Cider Creek: Cran Mango is a tropical and fruity cider made with 100% New York State apples, cranberry, and mango. Semi-sweet with low tartness. Gluten-free.",
        metadata={"type": "cider", "name": "Cider Creek Cran Mango", "abv": "6.9%", "gluten_free": True, "style": "Fruit Cider"}
    ),
    Document(
        page_content="Bolivar Black Gold is a dark ale with roasted malt complexity and smooth finish.",
        metadata={"type": "beer", "name": "Bolivar Black Gold", "abv": "5.8%", "ibu": 30, "style": "Dark Ale"}
    ),
    Document(
        page_content="Chocolate Peanut Butter Porter is brewed with chocolate and peanut butter. Slightly sweet, lightly roasted — perfect for chilly winter days.",
        metadata={"type": "beer", "name": "Chocolate Peanut Butter Porter", "abv": "4.5%", "ibu": 23, "style": "Porter"}
    ),
    Document(
        page_content="Cream Ale is smooth and easy-drinking with subtle malt sweetness, a hint of creaminess, and mild hop presence.",
        metadata={"type": "beer", "name": "Cream Ale", "abv": "5.1%", "ibu": 9, "style": "Cream Ale"}
    ),
    Document(
        page_content="Irish Red Ale is a malt-forward ale brewed with pale malt and roasted barley, giving it a beautiful red color and smooth taste.",
        metadata={"type": "beer", "name": "Irish Red Ale", "abv": "4.7%", "ibu": 24, "style": "Red Ale"}
    ),
    Document(
        page_content="Wells Oatmeal Stout is based on an English stout. Brewed with oats for body and silkiness, it's dark and rich with flavors of coffee and chocolate.",
        metadata={"type": "beer", "name": "Wells Oatmeal Stout", "abv": "4.5%", "ibu": 29, "style": "Stout"}
    ),
    Document(
        page_content="Sweet Summertime Kölsch Ale is bright and golden, with honeyed malt and floral hops. Bursting with pineapple and mango, crisp and refreshing.",
        metadata={"type": "beer", "name": "Sweet Summertime Kölsch Ale", "abv": "4.7%", "ibu": 22, "style": "Fruit Kölsch"}
    ),
    Document(
        page_content="Hard Seltzer is crisp, refreshing, and lightly sparkling. Offered in five flavors: Black Cherry, Mango, Wildberry, Coconut, and Citrus.",
        metadata={"type": "seltzer", "name": "Hard Seltzer", "abv": "4.7%", "flavors": ["Black Cherry","Mango","Wildberry","Coconut","Citrus"]}
    ),
    Document(
        page_content=(
            "Wellsville Brewing Company is a New York State farm brewery, "
            "offering a classy tasting room without the bar scene, featuring light share-plates, "
            "soup and sandwich specials, as well as tavern puzzles and board games for guests to enjoy."
        ),
        metadata={
            "type": "brewery_info",
            "category": "tasting_room",
            "features": ["share-plates", "soup specials", "sandwich specials", "board games", "tavern puzzles"]
        }
    ),
    Document(
        page_content=(
            "Located at 104 North Main Street, Wellsville, NY 14895, the brewery offers a cozy and family-friendly atmosphere."
        ),
        metadata={
            "type": "location",
            "address": "104 North Main Street, Wellsville, NY 14895",
            "phone": "(585) 296-3230"
        }
    ),
    Document(
        page_content=(
            "The Wellsville Brewing Company opened on October 6, 2018, in a fully renovated space on historic Main Street in Wellsville, New York."
        ),
        metadata={
            "type": "brewery_info",
            "category": "history",
            "opened": "2018-10-06"
        }
    ),
    Document(
        page_content=(
            "As of March 2025, the Mariotti family—Kera, Joe, Isabel, and JT Mariotti—"
            "are the proud new owners of the Wellsville Brewing Company."
        ),
        metadata={
            "type": "ownership",
            "owners": "Kera Mariotti, Joe Mariotti, Isabel Mariotti, JT Mariotti",
            "acquisition_date": "2025-03-07"
        }
    ),
    Document(
        page_content=(
            "New owners plan to operate under normal hours—Thursday & Friday from 3 pm to 9 pm, Saturday from 2 pm to 9 pm."
        ),
        metadata={
            "type": "hours",
            "Thursday": "15:00-21:00",
            "Friday": "15:00-21:00",
            "Saturday": "14:00-21:00"
        }
    )
]

# generate unique id's

i = 0
uuids = []

while i < len(documents):

    i += 1

    uuids.append(f"id{i}")

# add to database

vector_store.add_documents(documents=documents, ids=uuids)