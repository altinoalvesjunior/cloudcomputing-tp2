from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Carrega as regras do arquivo pickle no início
with open('rules.pkl', 'rb') as f:
    rules = pickle.load(f)

# Converte as colunas 'antecedents' e 'consequents' de volta para frozensets
rules['antecedents'] = rules['antecedents'].apply(frozenset)
rules['consequents'] = rules['consequents'].apply(frozenset)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    # Obtem as músicas favoritas do usuário da requisição
    user_songs = request.get_json(force=True)['songs']

    # Calcula as recomendações
    recommendations = set()
    for song in user_songs:
        # Encontra as regras que incluem a música atual nos antecedentes
        relevant_rules = rules[rules['antecedents'].apply(lambda x: song in x)]

        # Ordena as regras por confiança (decrescente) e lift (decrescente)
        relevant_rules = relevant_rules.sort_values(['confidence', 'lift'], ascending=[False, False])

        for _, rule in relevant_rules.iterrows():
            if rule['confidence'] > 0.7:
                # Adiciona todas as músicas dos consequentes que não estão nas músicas do usuário
                recommendations.update([song for song in rule['consequents'] if song not in user_songs])

            if len(recommendations) >= 5:  # Limita a 5 recomendações por música de entrada
                break

    # Converte o conjunto para lista
    recommendations = list(recommendations)

    # Retorna as recomendações como JSON
    response = {'songs': recommendations, 'version': '1.0', 'model_date': '2023-12-10'}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)