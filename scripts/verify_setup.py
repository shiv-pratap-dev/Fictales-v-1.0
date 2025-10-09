from pathlib import Path 
import os 
root = Path(__file__).resolve().parents[1] 
print('Project root:', root) 
print('Folders:') 
for d in ['input/kid_faces','input/pages','output/swapped_pages','output/pdfs','logs','tmp','src','scripts']: 
    print(' -', root.joinpath(d)) 
env = os.getenv('REPLICATE_API_TOKEN') 
print('\\nREPLICATE_API_TOKEN set in environment?' , bool(env)) 
print('\\nTo run verification: python scripts\\verify_setup.py') 
