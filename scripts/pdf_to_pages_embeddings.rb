require "csv"
require "json"
require "daru"
require "openai"
require "optparse"
require "pdf-reader"
require "tokenizers"
require "dotenv/load"


options = {}
OptionParser.new do |opts|
    opts.on("-pdf", "--pdf FILE", "PDF input file") do |file|
        options[:input] = file
      end
end.parse!

$OPENAI_API_KEY = ENV["OPENAI_API_KEY"]
$MODEL = "text-search-curie-doc-001"
$INPUT_FILENAME = options[:input]


def count_tokens(text) 
    """count the number of tokens in a string"""
    tokens = $tokenizer.encode(text).ids
    return tokens.length()
end

def extract_page(page_text, index)
    """
    Extract the text from the page
    """
    content = page_text.split.join(" ") # for some reason split and rejoin
    # puts "Page text: " + content
    output = ["Page " + index, content, count_tokens(content)] # pending count_tokens
    return output
end

def generate_tokenization()
    columns = ["title", "content", "tokens"]
    data = []
    $reader.pages.each_with_index do |page, idx|
        page_output = extract_page(page.text, idx.to_s)
        if page_output[2] < 2046 # removing pages with more than 2046 tokens (beacuse OpenAI api??)
            data.append(page_output)
        end
    end
    df = Daru::DataFrame.rows(data, order: columns)
    df.write_csv("#{$INPUT_FILENAME}.pages.csv")
    return df
end

def get_embedding(text, model)
    response = $openai_client.embeddings(
        parameters: {
            model: model,
            input: text
        }
    )
    return JSON.parse(response.body)["data"][0]["embedding"]
end

def compute_doc_embeddings(tokenization_df)
    embeddings = []
    tokenization_df.each_row do |row|
        content = row["content"]
        embeddings.append(get_embedding(content, $MODEL))
    end
    return embeddings
end

def generate_embeddings_csv(doc_embeddings)
    # CSV with exactly these named columns:
    # "title", "0", "1", ... up to the length of the embedding vectors.
    columns = ["title"].append(*(1...4097))

    data = doc_embeddings.map.with_index { |elem, i| elem.unshift("Page #{(i+1).to_s}") }
    Daru::DataFrame.rows(data, order: columns).write_csv("#{$INPUT_FILENAME}.embeddings.csv")
end


$reader = PDF::Reader.new($INPUT_FILENAME)
$tokenizer = Tokenizers.from_pretrained("gpt2")
$openai_client = OpenAI::Client.new(access_token: $OPENAI_API_KEY)

tokenization = generate_tokenization()
doc_embeddings = compute_doc_embeddings(tokenization)
generate_embeddings_csv(doc_embeddings)
