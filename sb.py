import cobra
from urllib.request import urlretrieve

# Download iJO1366 from BiGG
url = "http://bigg.ucsd.edu/static/models/iJO1366.xml"
urlretrieve(url, "iJO1366.xml")
print("Downloaded: iJO1366.xml")

# Verify it loads
model = cobra.io.read_sbml_model("iJO1366.xml")
print(f"Model: {model.id}, Reactions: {len(model.reactions)}")