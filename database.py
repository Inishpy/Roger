import sqlite3

def get_coordinates(location):
    # Connect to the SQLite database
    conn = sqlite3.connect('location_data.db')
    cursor = conn.cursor()

    # Query to retrieve the longitude and latitude based on the location field
    query = "SELECT longitude, latitude FROM locations WHERE location = ?"

    # Execute the query with the provided location
    cursor.execute(query, (location,))
    result = cursor.fetchone()

    # Close the connection
    conn.close()

    # Return the result as a tuple (longitude, latitude) or None if not found
    return result