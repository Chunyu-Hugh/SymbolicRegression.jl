"""
简单的LLM集成测试脚本

用于测试LLM API调用是否正常工作
"""

# 确保使用本地开发版本
const LOCAL_SR_PATH = joinpath(@__DIR__, "..", "src")
if !(LOCAL_SR_PATH in LOAD_PATH)
    pushfirst!(LOAD_PATH, LOCAL_SR_PATH)
end

using HTTP
using JSON

# LLM API 配置
const MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
const API_BASE_URL = "https://ollama.cs.odu.edu"
const API_KEY = "sk-704f0fe8734549c9b57ac09a69001ad7"

function test_llm_call()
    """测试LLM API调用"""
    url = "$API_BASE_URL/api/v1/chat/completions"
    
    headers = Dict(
        "Content-Type" => "application/json",
        "Authorization" => "Bearer $API_KEY"
    )
    
    payload = Dict(
        "model" => MODEL_NAME,
        "messages" => [
            Dict("role" => "user", "content" => "Hello, please introduce yourself in one sentence.")
        ],
        "stream" => false
    )
    
    println("正在测试LLM API调用...")
    println("模型: $MODEL_NAME")
    println("URL: $url")
    println()
    
    try
        response = HTTP.post(
            url,
            headers=headers,
            body=JSON.json(payload);
            readtimeout=30
        )
        
        println("响应状态码: $(response.status)")
        
        if response.status == 200
            body_str = String(response.body)
            result = JSON.parse(body_str)
            
            if haskey(result, "choices") && length(result["choices"]) > 0
                message = result["choices"][1]["message"]["content"]
                println("✓ LLM响应成功:")
                println(message)
                return true
            else
                println("✗ 响应中没有choices字段")
                println("完整响应: $body_str")
                return false
            end
        else
            println("✗ API调用失败")
            println("响应体: $(String(response.body))")
            return false
        end
    catch e
        println("✗ 请求失败: $e")
        return false
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("=" ^ 80)
    println("LLM API 测试")
    println("=" ^ 80)
    println()
    
    success = test_llm_call()
    
    println()
    println("=" ^ 80)
    if success
        println("✓ 测试通过！LLM API可以正常使用。")
    else
        println("✗ 测试失败！请检查网络连接和API配置。")
    end
    println("=" ^ 80)
end

