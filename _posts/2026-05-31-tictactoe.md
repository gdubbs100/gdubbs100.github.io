---
overview: "Exploring how LLMs play tic-tac-toe."
---
# Can LLMs play tic-tac-toe?
I've started trying expand my understanding of how LLMs operate and reason. I think it is pretty fascinating that the process of training a model to predict the next token in a sequence of natural language can lead to an emergent behaviour that seems to mimic logical reasoning pretty well. Moreover, if you have _enough_ data and the right type of data it seems they can even prove <a href="https://openai.com/index/model-disproves-discrete-geometry-conjecture/">mathematical theorems</a>.

On a few occasions when bored, I've asked whatever LLM I've been using to play tic-tac-toe. I've seen variable results, but recently I played one of the Qwen models and it played me to a draw. Admittedly, tic-tac-toe is a very simple game and a Google search will reveal many internet resources on how to play - which would suggest that LLMs have a lot of info about it in their training data. 

I thought I would try evaluating a few different models on how well they could play tic-tac-toe and if they played optimally. I also wanted to try and determine how much typical representations of the board affected their behaviour. Essentially, most representations of tic-tac-toe found on the internet would be designed to look something like this:

```
       X |   |  
      -----------
       O | X |  
      -----------
         |   |  
```

But if we replaced typical tokens with something else, say: 

```
       A )   )  
      -----------
       B ) A )  
      -----------
         )   )  
```

How well could an LLM recognise it? The symbols in the second board have a simple mapping to those of the first and any human could identify this and play an optimal game of tic-tac-toe on either board.

## Method
So I set up a simple tic-tac-toe environment with a textual state space that looked like the first state above. I set up some baseline algorithms - a random agent that plays a random move and an optimal agent which plays according to the <a href="https://en.wikipedia.org/wiki/Minimax">minmax algorithm</a>. 

I then selected some LLMs based on what I could access for free - either through cloud or to run locally on my laptop. In the end I mainly used <a href = "https://ollama.com/library/gemma4">Gemma4</a> with 31 billion parameters, but I also experimented with <a href="https://ollama.com/library/gpt-oss">GPT-oss</a> with 120 billion parameters (an open source GPT model from OpenAI), and Qwen 3.6 with 0.6 billion parameters. I ran Gemma4 and GPT-oss via Ollama cloud and Qwen 3.6 locally via HuggingFace.

I simulated games of tic-tac-toe and measured what proportion of moves were optimal. Thanks to Claude, I was also able to build an interactive dashboard where I could review the results of games and the thinking traces of models. This was very useful initially as I was able to debug errors / failures of the LLM and also get an insight into what was actually happening in the reasoning process.

I ran a few different experiments:
1. A tournament amongst random, minmax (alphabeta), Gemma4, GPT-oss and Qwen to evaluate how well the language models performed on standard tic-tac-toe.
2. Various games (Gemma4 vs alphabeta) with different symbol representations to assess how the LLM would respond to unusual symbols and representations of the board
3. A 4x4 tic-tac-toe game to see how well LLMs could perform in a (slightly) more complex game

I had limited resources as Ollama only offers so many free calls to cloud models per session. As a result, I often ended up focussing on the results for Gemma 4, as this model performed the best overall.

## Prompt design
I went through a few iterations of prompts. My intention was to design a prompt that would give enough information to the LLM so that it could play the game, but I didn't want to tell it _how_ to play. I wanted it to figure out that for itself. For example, I didn't want to tell it how to read the board, I wanted it to be able to look at the state and figure out what each of the symbols meant. I also didn't want to tell it that it was playing tic-tac-toe, as that comes with a whole bunch of knowledge it could query from its training data - instead, I just told it the rules. In most cases, the LLM figured out it was tic-tac-toe and mentioned as such in its reasoning traces anyway.

In my final prompt I wrote this:

```
      You are playing a game on a 3x3 grid. The cells are numbered 0-8 in
      row-major order, starting at the top-left (0) and going left-to-right,
      top-to-bottom, so the bottom-right cell is 8:

       0 | 1 | 2
      -----------
       3 | 4 | 5
      -----------
       6 | 7 | 8

      Two players take turns placing their mark in an empty cell. The first player to get 3 of their
      own marks in a line — horizontally, vertically, or diagonally — wins. If
      every cell is filled with no such line, the game is a draw.
      
      Respond with a single integer (the cell number) only. No explanation.
```

I avoided mentioning the name of the game, but gave it information about the board setup, allowable moves and objectives.

## Experiments

### Experiment 1: A rocky start with Qwen 0.6B
discuss initial tournament. Re-run with Qwen.
Findings are
- most of the time larger models play optimally.
- Qwen with 0.6 params is a non-starter (but it is likely that larger models will work) as it just struggles

### Experiment 2: Gemma4 seems fairly robust to representation changes
Run games against alphabeta with representation changes. Generally, Gemma4 seems to play optimally although note the greater token usage as things get more difficult. Also analyse some of the quirks with the thinking traces. Are they really thinking? is it sensible?

### Experiment 3: 4x4 Tic-tac-toe is also no issue
Increasing the complexity to 4x4 Tic-tac-toe does not seem to throw LLMs off - but it is likely that there are numerous games in training data.

Compare token useage?
Could you try 5x5?

Ideally we would go to more complex games but my current set up doesn't allow for that (e.g. how to calculate optimality)

## How rational are the LLMs?
Try to run Gemma locally, get logprobs and plot against optimal actions. Look at kendall's tau or something.