@app.route('/structure', methods=['GET'])
def get_structure():
    try:
        with open('Output/structure.json', 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        return jsonify({'error': str(e)})