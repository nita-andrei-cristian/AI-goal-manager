#include "llama.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>

std::random_device rd;
std::mt19937 gen(rd()); // Mersenne Twister engine

std::string load_file(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Path not found");
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

inline void ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
}

enum AI_ROLE {
  CHAT,
  CREATE_PLAN,
  DETECTOR,
};

struct Goal {

  double t;
  std::string name;

  Goal *children[10];
  Goal *parent;
  char childrensize;

  Goal(std::string s, double days) {
    t = days;
    name = s;
    parent = nullptr;

    childrensize = 0;
  }

  ~Goal() {
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
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f,
n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");

    return 0;
}


*/

class AI {
private:
  static char next_id;
  char id;

public:
  llama_model *model;
  AI_ROLE role;

  llama_model_params params;
  const llama_vocab *vocab;
  llama_sampler *smpl;
  const int n_predict = 64;
  llama_context *ctx;
  std::string gbnf;

  AI(AI_ROLE _role, std::string path) {
    role = _role;

    // set params to default if none
    params = llama_model_default_params();
    params.n_gpu_layers = 99; // force GPU as much as possible

    // load model from file
    model = llama_model_load_from_file(path.c_str(), params);
    if (model == NULL) {
      fprintf(stderr, "ERROR Unable to load model %s : %s", &id, __func__);
    }

    // set vocab
    vocab = llama_model_get_vocab(model);

    // set grammar
    gbnf = get_grammar();

    // sampler
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    smpl = llama_sampler_chain_init(sparams);

    // increment id for next agent
    id = next_id;
    next_id++;

    printf("AGENT %d %d is ready.\n", id, role);
  }

  std::string get_grammar() {
    if (role == CREATE_PLAN) {
      return R"(
root ::= item+

# Excludes various line break characters
item ::= "- " [^\r\n\x0b\x0c\x85\u2028\u2029]+ "\n"
    )";
    }
    return "";
  }

  std::string get_prompt(std::string user_prompt) {

    // format for current model (gemma 2).

    std::string prompt;
    prompt += "<start_of_turn>user";
    if (role == DETECTOR) {
      prompt = "You are a model that's main role is to detect logical "
               "errors.";
    } else if (role == CREATE_PLAN) {
      prompt =
          "You are a creative model who is in charge for "
          "breaking a human.  "
          "goal into logical subgoals. Keep your response short, "
          "mechanical and accurate. No titles, no fancy formating, just "
          "steps. Your answers are straight forward and not vague, they "
          "are meant to guide a lost person. Your answer will be processed in "
          "list form according do youi grammar, so make sure to focus on the "
          "items themselfs, only steps are relevant, not any extra info. The "
          "steps are named like book chapters to spark interest (example : "
          "'Provide sunlight' -> 'Providing the necessary sunlight'). You "
          "will receive the details starting with the parent goals of the "
          "current to make sure you know the context. You will only split the "
          "present one. Provide as many steps as you want (max 7). Make sure "
          "steps are "
          "not repetitive!";
    }
    prompt += user_prompt;
    prompt += "<end_of_turn>";
    prompt += "<start_of_turn>model";
    prompt += "<end_of_turn>";
    return prompt;
  }

  std::string run(std::string user_prompt, int min_tokens = 150) {
    std::string output = "";
    std::string prompt = get_prompt(user_prompt);

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                         NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                       prompt_tokens.data(), prompt_tokens.size(), true,
                       true) < 0) {
      fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
      return "ERROR (see in logs)";
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = false;

    ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
      fprintf(stderr, "%s: error: failed to create the llama_context\n",
              __func__);
      return "ERROR (see in logs)";
    }

    // initialize the sampler
    llama_sampler_chain_add(
        smpl, llama_sampler_init_grammar(vocab, gbnf.c_str(), "root"));
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // prepare batch
    llama_batch batch =
        llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    if (llama_model_has_encoder(model)) {
      if (llama_encode(ctx, batch)) {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        return "ERROR (see in logs)";
      }

      llama_token decoder_start_token_id =
          llama_model_decoder_start_token(model);
      if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
        decoder_start_token_id = llama_vocab_bos(vocab);
      }

      batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }

    // main loop

    int n_decode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict;) {
      // evaluate the current batch with the transformer model
      if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
        return "ERROR (see in logs)";
      }

      n_pos += batch.n_tokens;

      // sample the next token
      {
        new_token_id = llama_sampler_sample(smpl, ctx, -1);
        // llama_sampler_accept(smpl, new_token_id);

        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0,
                                     true);

        if (n < 0) {
          fprintf(stderr, "%s: error: failed to convert token to piece\n",
                  __func__);
          return "ERROR (see in logs)";
        }

        std::string s(buf, n);
        output += s;

        if (output.find("<eos>") != std::string::npos) {
          output = "\n";
        }

        std::string end_token = "<end_of_turn>";
        if (output.find(end_token) != std::string::npos &&
            n_decode < min_tokens) {
          output.erase(output.length() - end_token.length(),
                       end_token.length());
          break;
        }
        fflush(stdout);

        // prepare the next batch with the sampled token
        batch = llama_batch_get_one(&new_token_id, 1);

        n_decode += 1;
      }
    }

    return output;
  }

  ~AI() {
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
  }

  std::string computeSubGoal(Goal g, int i) { return "test goal"; }
  bool test(Goal g, std::string name) { return true; };

  char readID() const { return id; }
};
char AI::next_id = 0;

std::string parent_chain(Goal &goal) {
  std::string output = "";

  Goal *parent = goal.parent;
  while (parent != nullptr) {
    output = parent->name + "->" + output;
    parent = parent->parent;
    if (parent->parent == parent)
      break;
  }

  return output;
}
void SPLIT_GOAL(Goal &goal, AI &creator) {
  std::cout << "PROMPT : "
            << (parent_chain(goal) + " | you will split : How to " + goal.name)
            << "\n";
  std::string response = creator.run(parent_chain(goal) +
                                     " | you will split : How to " + goal.name);
  std::istringstream stream(response);

  std::string line;

  int index = 0;
  while (std::getline(stream, line)) {
    line.erase(0, 2); // removes the " -"
    ltrim(line);
    // std::cout << "CREATING GOAL\n" << line << "\n";

    if (line.empty())
      continue;

    Goal *child = new Goal(line, 20);
    if (&goal != child)
      child->parent = &goal;
    goal.children[index++] = child;
  }

  goal.childrensize = index;
}

int main() {

  ggml_backend_load_all();

  llama_log_set(
      [](enum ggml_log_level level, const char *text, void * /* user_data */) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
          fprintf(stderr, "%s", text);
        }
      },
      nullptr);

  auto agent1 =
      AI(CREATE_PLAN,
         "/home/nita/dev/cpp/AI-goal-manager/models/gemma-2-9b-it-Q6_K.gguf");

  system("clear");
  std::string goalname;
  std::cout << "Create a goal : ";

  getline(std::cin, goalname);
  std::cout << "\nGoal created successfully!\n";
  Goal goal(goalname, 100);

  int option;
  std::cout << '\n';
  while (true) {

    std::cout << "\nCurrent Goal :" << goal.name << '\n';
    std::cout << "Enter an option:    ";

    std::cout << "[0] exit         ";
    std::cout << "[1] split goal\nYou choose : ";

    std::cin >> option;

    if (option == 0)
      break;
    if (option == 1) {
      SPLIT_GOAL(goal, agent1);

      system("clear");

      std::cout << "\n Divisions \n\n";
      for (int i = 0; i < goal.childrensize; i++) {
        std::cout << "[" << (i + 1) << "] : " << goal.children[i]->name << "\n";
      }
      std::cout << "\n Choose a next goal: \n\n";

      std::cin >> option;
      goal = *goal.children[option - 1];
      std::cout << "You have chosen : " << goal.name;

      system("clear");
    }
  }

  return 0;
}
