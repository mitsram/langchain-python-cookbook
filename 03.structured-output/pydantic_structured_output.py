from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Review(BaseModel):
    key_themes: list[str] = Field(description="Key themes to focus on during the trip")
    summary: str = Field(description="A brief summary of the travel plan")
    sentiment: Literal["positive", "neutral", "negative"] = Field(description="Overall sentiment of the travel plan")
    name: Optional[str] = Field(description="Name of the reviewer, if available")

structured_model = model.with_structured_output(Review, strict=True)

prompt = """Tokyo, Japan stands as one of the world's most captivating destinations, where ancient traditions seamlessly blend with cutting-edge technology and modern innovation. This sprawling metropolis of over 14 million people offers an extraordinary tapestry of experiences, from serene temple gardens and traditional tea ceremonies to neon-lit entertainment districts and Michelin-starred restaurants. The city's efficient public transportation system makes it remarkably easy to explore diverse neighborhoods, each with its own distinct character and charm.

The cultural richness of Tokyo is unparalleled, encompassing everything from the historic Senso-ji Temple in Asakusa to the futuristic architecture of Odaiba. Visitors can witness the disciplined chaos of the Shibuya crossing, find tranquility in the Meiji Shrine's forested grounds, experience the otaku culture of Akihabara's electronics district, or shop in the luxury boutiques of Ginza. The city's culinary scene is legendary, offering not just sushi and ramen, but an entire universe of regional Japanese cuisine, street food, izakaya experiences, and innovative fusion restaurants that have earned Tokyo more Michelin stars than any other city in the world.

For a traveler seeking an immersive 6-day experience in Tokyo with a budget of $3000, the possibilities are both exciting and carefully balanced. This budget allows for comfortable mid-range accommodation, unlimited exploration via the incredibly efficient train and subway system, a mix of fine dining and casual eateries, and access to both popular attractions and hidden local favorites. The journey should capture Tokyo's multifaceted identity—its reverence for tradition, its embrace of the future, its obsession with quality and detail, and its warm hospitality.

Please create a comprehensive travel plan that identifies the key themes defining this Tokyo adventure, such as culinary exploration, cultural immersion, modern technology experiences, traditional arts, or urban discovery. Include a curated list of activities that showcase Tokyo's diversity: perhaps early morning visits to Tsukiji Outer Market, exploring the digital art installations at teamLab Borderless, experiencing a traditional tea ceremony, visiting historic temples and shrines, discovering quirky themed cafes, enjoying cherry blossoms or seasonal festivals, taking day trips to nearby areas, and experiencing Tokyo's legendary nightlife in districts like Shinjuku or Roppongi.

Provide a realistic budget breakdown covering accommodation in a well-located hotel or ryokan, transportation including a JR Pass or IC card for unlimited metro travel, dining experiences ranging from conveyor belt sushi to kaiseki dinners, activity fees and entrance tickets, and shopping or souvenir budget. Also outline the compelling pros of this itinerary—what makes it special and why Tokyo is worth visiting—as well as honest cons such as language barriers, potential culture shock, the fast pace of the city, budget constraints, or physical demands of extensive walking and navigating crowded areas.

Reviewed by Mitchell

"""


result = structured_model.invoke(prompt)
print(result)