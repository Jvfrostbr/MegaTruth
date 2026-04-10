#!/usr/bin/env python3
"""
Script para testar se a chave OpenRouter API está válida e ativa.
"""

import os
import requests
from dotenv import load_dotenv

# Carrega .env
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("❌ ERRO: OPENROUTER_API_KEY não definida no .env")
    exit(1)

print(f"🔑 Chave encontrada: {api_key[:20]}...{api_key[-10:]}")
print("\n" + "="*60)
print("Testando conectividade com OpenRouter...")
print("="*60 + "\n")

# Teste 1: Verificar se a chave é válida (request simples)
print("[1/3] Testando autenticação da chave...")
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Faz um teste leve - obtém lista de modelos (não consome créditos)
try:
    response = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers=headers,
        timeout=10
    )
    
    if response.status_code == 200:
        print("✅ Autenticação bem-sucedida!")
        print(f"   Status: {response.status_code}")
        models = response.json().get("data", [])
        print(f"   Total de modelos disponíveis: {len(models)}")
    elif response.status_code == 401:
        print("❌ ERRO 401: Chave inválida ou expirada!")
        print(f"   Resposta: {response.json()}")
        exit(1)
    elif response.status_code == 429:
        print("⚠️ ERRO 429: Rate limit atingido - tente novamente em alguns minutos")
        exit(1)
    elif response.status_code == 503:
        print("⚠️ ERRO 503: Servidor OpenRouter fora do ar temporariamente")
        exit(1)
    else:
        print(f"❌ Erro inesperado: {response.status_code}")
        print(f"   Resposta: {response.text}")
        exit(1)
        
except requests.exceptions.Timeout:
    print("❌ Timeout: Não conseguiu conectar ao servidor (sem internet ou servidor down)")
    exit(1)
except requests.exceptions.ConnectionError:
    print("❌ Erro de conexão: Verifique sua internet")
    exit(1)
except Exception as e:
    print(f"❌ Erro geral: {e}")
    exit(1)

# Teste 2: Verificar créditos/saldo
print("\n[2/3] Verificando saldo de créditos...")
try:
    response = requests.get(
        "https://openrouter.ai/api/v1/user",
        headers=headers,
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Informações da conta obtidas:")
        print(f"   Saldo: ${data.get('balance', 'N/A')}")
        print(f"   Uso: ${data.get('usage', 'N/A')}")
    else:
        print(f"⚠️ Não foi possível obter saldo: {response.status_code}")
        
except Exception as e:
    print(f"⚠️ Erro ao obter saldo: {e}")

# Teste 3: Testar com um modelo específico (teste real com baixo custo)
print("\n[3/3] Testando requisição com o modelo Nemotron...")
try:
    payload = {
        "model": "nvidia/nemotron-nano-12b-v2-vl:free",
        "messages": [
            {
                "role": "user",
                "content": "Responda com apenas uma palavra: FUNCIONA"
            }
        ],
        "max_tokens": 10
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        resposta = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"✅ Modelo respondeu: '{resposta}'")
        print(f"   Tokens usados: {result.get('usage', {}).get('total_tokens', 'N/A')}")
    elif response.status_code == 503:
        print("⚠️ ERRO 503: Servidor OpenRouter/Nemotron fora do ar no momento")
        print("   Tente novamente em alguns minutos")
    elif response.status_code == 429:
        print("⚠️ ERRO 429: Rate limit atingido")
    elif response.status_code == 401:
        print("❌ ERRO 401: Chave inválida")
    else:
        print(f"❌ Erro na requisição: {response.status_code}")
        print(f"   Resposta: {response.json()}")
        
except requests.exceptions.Timeout:
    print("⚠️ Timeout na requisição (servidor pode estar lento)")
except Exception as e:
    print(f"❌ Erro ao testar modelo: {e}")

print("\n" + "="*60)
print("✅ Testes concluídos!")
print("="*60)
