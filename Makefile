.PHONY: embed interface

embed: find_meme/embedding.py 
	python3 find_meme/embedding.py

interface: find_meme/interface.py
	python3 find_meme/interface.py