#include <string>
#include <iostream>
#include <random>
#include "llama.h"

std::random_device rd;
std::mt19937 gen(rd()); // Mersenne Twister engine

enum task
{
    CHAT,
    CREATE_PLAN,
    DETECTOR,
};

struct Goal
{

    double t;
    std::string name;

    Goal *children[10];
    char childrensize;

    Goal(std::string s, double days)
    {
        t = days;
        name = s;

        childrensize = 0;
    }

    ~Goal()
    {
        for (int i = 0; i < childrensize; i++)
            delete children[i];
    }

};

/*
BASIC AI USAGE

#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    // prepare a batch for the prompt


    // main loop

    printf("\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");

    return 0;
}

 
*/

class AI
{
    // Placeholder for AI functionalities

    int ID;
    task T;

    llama_model* model;
    llama_model_params model_params;
    const llama_vocab *vocab;

    std::string model_path;
    std::string prompt = "Hello my name is";
    int ngl = 99;
    int n_predict = 32;

    void load_model(std::string model_path, llama_model_params model_params)
    {

        model = llama_model_load_from_file(model_path.c_str(), model_params);
        vocab = llama_model_get_vocab(model);

        if (model == NULL)
        {
            fprintf(stderr, "%s: error: unable to load model\n", __func__);
        }
    }

    void unload_model(llama_sampler *smpl, llama_context *ctx, llama_model *model)
    {
        llama_perf_sampler_print(smpl);
        llama_perf_context_print(ctx);

        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
    }

    void set_params(int ngl, int n_predict)
    {

        this->ngl = ngl;
        this->n_predict = n_predict;

        model_params = llama_model_default_params();
        model_params.n_gpu_layers = ngl;
    }

    llama_context_params context_init(llama_context *&ctx, const std::vector<llama_token> &prompt_tokens)
    {
        // initialize the context

        llama_context_params ctx_params = llama_context_default_params();
        // n_ctx is the context size
        ctx_params.n_ctx = prompt_tokens.size() + n_predict - 1;
        // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
        ctx_params.n_batch = prompt_tokens.size();
        // enable performance counters
        ctx_params.no_perf = false;

        llama_context *ctx = llama_init_from_model(model, ctx_params);

        if (ctx == NULL)
        {
            fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
            return ctx_params;
        }

        return ctx_params;
    }

    llama_batch get_batch(std::vector<llama_token> &prompt_tokens, llama_context *ctx, const llama_vocab *vocab)
    {
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

        if (llama_model_has_encoder(model))
        {
            if (llama_encode(ctx, batch))
            {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return batch;
            }

            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == LLAMA_TOKEN_NULL)
            {
                decoder_start_token_id = llama_vocab_bos(vocab);
            }

            batch = llama_batch_get_one(&decoder_start_token_id, 1);
        }

        return batch;
    }

    llama_sampler* sampler_init()
    {

        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        llama_sampler *smpl = llama_sampler_chain_init(sparams);

        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        return smpl;
    }

    std::vector<llama_token> tokenize_prompt(std::string prompt, const llama_vocab *vocab, std::vector<llama_token> &prompt_tokens)
    {
        // tokenize the prompt

        // find the number of tokens in the prompt
        const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
        // allocate space for the tokens and tokenize the prompt
        std::vector<llama_token> prompt_tokens(n_prompt);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0)
        {
            fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        }
        return prompt_tokens;
    }

    std::vector<char[128]> MainLoopBuffers(llama_batch &batch, llama_context *ctx, llama_sampler *smpl, const llama_vocab *vocab, int n_prompt)
    {
        int n_decode = 0;
        llama_token new_token_id;

        std::vector<char[128]> char_buffers;

        for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict;)
        {
            // evaluate the current batch with the transformer model
            if (llama_decode(ctx, batch))
            {
                fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
                return char_buffers;
            }

            n_pos += batch.n_tokens;

            // sample the next token
            {
                new_token_id = llama_sampler_sample(smpl, ctx, -1);

                // is it an end of generation?
                if (llama_vocab_is_eog(vocab, new_token_id))
                {
                    break;
                }


                char_buffers.push_back({});
                int n = llama_token_to_piece(vocab, new_token_id, char_buffers.back(), sizeof(char_buffers.back()), 0, true);

                if (n < 0)
                {
                    fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                    return char_buffers;
                }

                // prepare the next batch with the sampled token
                batch = llama_batch_get_one(&new_token_id, 1);

                n_decode += 1;
            }
        }

        return char_buffers;
    }

    std::string MainLoop(llama_batch &batch, llama_context *ctx, llama_sampler *smpl, const llama_vocab *vocab, int n_prompt)
    {
        int n_decode = 0;
        llama_token new_token_id;

        std::string str;

        for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict;)
        {
            // evaluate the current batch with the transformer model
            if (llama_decode(ctx, batch))
            {
                fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
                return "";
            }

            n_pos += batch.n_tokens;

            // sample the next token
            {
                new_token_id = llama_sampler_sample(smpl, ctx, -1);

                // is it an end of generation?
                if (llama_vocab_is_eog(vocab, new_token_id))
                {
                    break;
                }


                char buf[128];
                int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
                str += std::string(buf, n);

                if (n < 0)
                {
                    fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                    return "";
                }

                // prepare the next batch with the sampled token
                batch = llama_batch_get_one(&new_token_id, 1);

                n_decode += 1;
            }
        }

        return str;

    }

    std::string run(std::string prompt)
    {
        // Load model
        load_model(model_path, model_params);

        // Tokenize prompt
        std::vector<llama_token> prompt_tokens;
        tokenize_prompt(prompt, vocab, prompt_tokens);

        // Initialize context
        llama_context *ctx;
        context_init(ctx, prompt_tokens);

        // Get batch
        llama_batch batch = get_batch(prompt_tokens, ctx, vocab);

        // Initialize sampler
        llama_sampler *smpl = sampler_init();

        // Main loop
        std::string output = MainLoop(batch, ctx, smpl, vocab, prompt_tokens.size());

        // Unload model and free resources
        unload_model(smpl, ctx, model);

        return output;
    }

    std::vector<char[128]> run_buffered_result()
    {
        // Load model
        load_model(model_path, model_params);

        // Tokenize prompt
        std::vector<llama_token> prompt_tokens;
        tokenize_prompt(prompt, vocab, prompt_tokens);

        // Initialize context
        llama_context *ctx;
        context_init(ctx, prompt_tokens);

        // Get batch
        llama_batch batch = get_batch(prompt_tokens, ctx, vocab);

        // Initialize sampler
        llama_sampler *smpl = sampler_init();

        // Main loop
        std::vector<char[128]> char_buffers = MainLoopBuffers(batch, ctx, smpl, vocab, prompt_tokens.size());

        // Unload model and free resources
        unload_model(smpl, ctx, model);

        return char_buffers;
    }

public:
    AI(int id, task t)
    {
        ID = id;
        T = t;

        model_path = "";
        prompt = "Hello my name is";
        set_params(99, 32);
    }
    ~AI(){
        llama_model_free(model);
    }

    bool test(Goal &g, std::string data)
    {
        // Placeholder for testing logic
        std::cout << "Detector AI testing goal: " << data << "\n";

        //randomize result

        std::uniform_int_distribution<> dist(0, 4);

        // Generate a random number
        int randomNum = dist(gen);

        return randomNum != 0; // 80% chance to pass
    }

    std::string readID()
    {
        return std::to_string(ID);
    }

    std::string computeSubGoal(Goal &g, int i)
    {
        // Placeholder for subgoal computation logic
        return g.name + " Subgoal " + std::to_string(i);
    }
};

void COMPUTE_SPLIT(Goal &g, AI &creator, int n = 2)
{

    for (g.childrensize = 0; g.childrensize < n; g.childrensize++)
    {
        auto name = creator.computeSubGoal(g, g.childrensize);
        g.children[g.childrensize] = new Goal(name + " -> " + std::to_string(g.childrensize), g.t / n);
    }
}

void PERFORM_SPLIT(Goal &g, int n, AI &creator, AI &detector, int depth = 0)
{
    std::cout << "AI " << creator.readID() << " is splitting the goal \"" << g.name << "\" into " << n << " subgoals.\n";
    COMPUTE_SPLIT(g, creator, n);

    for (int i = 0; i < n; i++)
    {
        auto child = g.children[i];
        auto data = child->name;

        bool result = detector.test(g, data);
        if (result)
        {
            std::cout << "Goal \"" << child->name << "\" passed the detector test.\n";
        }
        else if (!result && depth < 10)
        { // Limit recursion depth
            std::cout << "Goal \"" << child->name << "\" failed the detector test.\n";

            PERFORM_SPLIT(g, n, creator, detector, depth + 1);
            return;
        }
        else
        {
            std::cout << "Goal \"" << child->name << "\" failed the detector test and maximum depth reached. Aborting further splits.\n";
        }
    }
}

int main()
{

    ggml_backend_load_all();

    Goal goal("Finish app", 100);

    auto agent2 = AI(1, CREATE_PLAN);
    auto agent1 = AI(0, DETECTOR);

    int n = 0;
    std::cout << "Into how many subgoals do you want to split your goal? ";
    std::cin >> n;

    PERFORM_SPLIT(goal, n, agent2, agent1);

    std::cout << "\nFinal subgoals:\n";
    for (int i = 0; i < n; i++)
        std::cout << (*goal.children[i]).name << " : " << std::to_string((*goal.children[i]).t) << "\n";
    

    int x = 0;
    std::cin >> x;

    return 0;
}