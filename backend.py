from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import json
from sklearn.preprocessing import MinMaxScaler
import joblib
import re
import locale

class RealEstatePredictor:
    def __init__(self, model_path='real_estate_model.joblib'):
        """
        Carregar modelo salvo e preparar para predições
        
        Args:
            model_path (str): Caminho para o modelo salvo
        """
        # Carregar modelo completo
        self.pipeline = joblib.load(model_path)
        
    def predict(self, features):
        """
        Fazer predição para novos dados
        
        Args:
            features (dict or list): Features para predição
        
        Returns:
            float: Preço predito
        """
        # Converter para DataFrame se for um dicionário
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        # Colunas esperadas
        expected_columns = [
            'Área', 'Quartos', 'Banheiros', 'Garagens', 
            'area_por_quarto', 'densidade_ocupacao', 'area_por_garagem', 
            'Lat', 'Long'
        ]
        
        # Adicionar features sintéticas se não existirem
        if 'area_por_quarto' not in features.columns:
            features['area_por_quarto'] = features['Área'] / (features['Quartos'] + 1)
        
        if 'densidade_ocupacao' not in features.columns:
            features['densidade_ocupacao'] = features['Quartos'] * features['Banheiros']
        
        if 'area_por_garagem' not in features.columns:
            features['area_por_garagem'] = features['Área'] / (features['Garagens'] + 1)
        
        # Garantir ordem das colunas
        features = features[expected_columns]
        
        # Fazer predição
        prediction = self.pipeline.predict(features)
        
        return float(prediction[0])

def save_model(predictor, save_path='real_estate_model.joblib'):
    """
    Salvar modelo treinado
    
    Args:
        predictor (AdvancedRealEstatePredictor): Modelo treinado
        save_path (str): Caminho para salvar o modelo
    """
    joblib.dump(predictor.best_model, save_path)
    print(f"Modelo salvo em {save_path}")

# Script para salvar o modelo após treinamento
def main():
    from grad_boosting_gridsearch import AdvancedRealEstatePredictor
    
    # Treinar modelo
    predictor = AdvancedRealEstatePredictor('data_total.csv')
    predictor.train_and_evaluate()
    
    # Salvar modelo
    save_model(predictor)

app = Flask(__name__)

# Carrega os dados de um arquivo JSON
def carregar_dados():
    try:
        with open("imoveis_com_coordenadas_unificados.html", "r", encoding="utf-8") as f:
            data = json.load(f)  # Carrega o arquivo JSON
        return pd.DataFrame(data)  # Converte para um DataFrame
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return pd.DataFrame([])  # Retorna DataFrame vazio em caso de erro

def tratar_faixa(valor):
    # Verifica se o valor é uma string com faixa (exemplo: '2 - 3')
    if isinstance(valor, str) and ' - ' in valor:
        partes = valor.split(' - ')
        return (int(partes[0]) + int(partes[1])) / 2  # Retorna a média
    
    # Verifica se o valor é no formato 'X ou mais' (exemplo: '5 ou mais')
    elif isinstance(valor, str) and ' ou mais' in valor:
        partes = valor.split(' ou mais')
        return int(partes[0])  # Retorna o valor mínimo da faixa
    
    # Se for um valor numérico, retorna como float
    else:
        return float(valor)

# Exemplo de função para limpar e converter o preço
def limpar_preco(preco):
    if isinstance(preco, str):
        # Remove "R$" e os pontos de milhares, depois converte para float
        preco = preco.replace('R$', '').replace('.', '').replace(',', '.')
        return float(preco)
    return preco

@app.route('/')
def index():
    # Renderiza a página inicial (HTML)
    return render_template("index.html")

def limpar_area(area):
    # Remove a unidade 'm²' (caso exista) e converte para número
    area = re.sub(r'\s*m²$', '', str(area))  # Remove "m²" no final da string
    try:
        return float(area)
    except ValueError:
        return 0.0  # Retorna 0 caso a conversão falhe

@app.route('/buscar', methods=['POST'])
def buscar_imoveis():
    # Carregar os dados do arquivo
    imoveis = carregar_dados()

    imoveis = imoveis[imoveis['Preço'].notna()]

    # Preenche valores ausentes para Quartos, Banheiros, Garagens e Área
    imoveis[['Quartos', 'Banheiros', 'Garagens', 'Área']] = imoveis[['Quartos', 'Banheiros', 'Garagens', 'Área']].fillna(0)

    imoveis['Quartos'] = imoveis['Quartos'].apply(tratar_faixa)
    imoveis['Banheiros'] = imoveis['Banheiros'].apply(tratar_faixa)
    imoveis['Garagens'] = imoveis['Garagens'].apply(tratar_faixa)
    imoveis['Preço'] = imoveis['Preço'].apply(limpar_preco)
    
    # Limpa a área removendo "m²" e convertendo para número
    imoveis['Área'] = imoveis['Área'].apply(limpar_area)

    if imoveis.empty:
        return jsonify({"erro": "Nenhum dado disponível para buscar imóveis."}), 500

    # Recebe os dados do usuário
    dados_usuario = request.json

    # Verifica se os dados do usuário têm os campos necessários
    try:
        user_input = pd.DataFrame([{
            "Quartos": dados_usuario["Quartos"],
            "Banheiros": dados_usuario["Banheiros"],
            "Garagens": dados_usuario["Garagens"],
            "Area": dados_usuario["Area"]
        }])
    except KeyError as e:
        return jsonify({"erro": f"Campo necessário ausente: {e}"}), 400
    
    imoveis['Distancia'] = 0.0
    
    # Calcula a distância euclidiana com base nos critérios
    try:
        imoveis['Distancia'] = euclidean_distances(
            imoveis[['Quartos', 'Banheiros', 'Garagens', 'Área']],
            user_input[['Quartos', 'Banheiros', 'Garagens', 'Area']]
        ).flatten()
    except Exception as e:
        return jsonify({"erro": f"Erro ao calcular a distância: {e}"}), 500

    # Ordena pelo menor preço e distância
    resultados = imoveis.sort_values(['Distancia']).head(50)
    resultados = resultados.sort_values(['Preço'], ascending=True)

    return jsonify(resultados.to_dict(orient='records'))

predictor = RealEstatePredictor()

@app.route('/predict', methods=['POST'])
def predict_price():
    """
    Endpoint para predição de preço de imóvel
    
    Entrada esperada no JSON:
    {
        "Área": 100,
        "Quartos": 3,
        "Banheiros": 2,
        "Garagens": 1,
        "Lat": -23.5505,
        "Long": -46.6333
    }
    """
    try:
        # Receber dados do request
        data = request.json
        
        # Validar dados de entrada
        required_fields = ['Área', 'Quartos', 'Banheiros', 'Garagens', 'Lat', 'Long']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Campo '{field}' é obrigatório"}), 400
        
        # Fazer predição
        predicted_price = predictor.predict(data)
        locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
        
        return jsonify({
            "predicted_price": locale.currency(predicted_price, grouping=True),
            "message": "Preço do imóvel predito com sucesso"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
