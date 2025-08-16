import time
from helper import *

if __name__ == "__main__":
#     # Assuming 'grid_tiles' is the 2D list of tile images from Step 2
# # and 'templates' is the dictionary from Step 3
    # delay for 1 second to allow user to switch to the game window
    time.sleep(1)
    grid_state = []
    image = get_image(550, 230, 825, 825)  # Example coordinates
    grid_tiles = split_grid_into_tiles(image, rows=3, cols=3, print=False)
    template_folder = 'templates'  # Path to your templates folder
    templates = load_templates(template_folder)
    for r in range(len(grid_tiles)):
        row_state = []
        for c in range(len(grid_tiles[r])):
            tile_image = grid_tiles[r][c]
            # Identify the content of the tile
            tile_content = identify_tile(tile_image, templates, threshold=0.8)
            tile_score = read_number_from_icon(tile_image)
            if tile_content is not None:
                print(f"Tile at ({r}, {c}) identified as: {tile_content} with score: {tile_score}")
            else:
                print(f"Tile at ({r}, {c}) could not be identified.")
            row_state.append({'content': tile_content, 'score': tile_score})
        grid_state.append(row_state)

    # Print the final grid state
    for row in grid_state:
        print(row)