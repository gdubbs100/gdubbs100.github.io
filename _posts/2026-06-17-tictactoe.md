---
overview: "Exploring how LLMs play tic-tac-toe."
---
# Can LLMs play tic-tac-toe?
I've started trying expand my understanding of how LLMs operate and reason. I think it is pretty fascinating that the process of training a model to predict the next token in a sequence of natural language can lead to an emergent behaviour that seems to mimic logical reasoning pretty well. Moreover, if you have _enough_ data and the right type of data it seems they can even prove <a href="https://openai.com/index/model-disproves-discrete-geometry-conjecture/">mathematical theorems</a>.

On a few occasions when bored, I've asked whatever LLM I've been using to play tic-tac-toe. I've seen variable results. Recently, however, I played one of the Qwen models and it played me to a draw. I was surprised, and I wanted to understand this a bit more. Admittedly, tic-tac-toe is a very simple game and a Google search will reveal many internet resources on how to play - which would suggest that LLMs have a lot of info about it in their training data. Even so, I decided to dig a little deeper and see what I could glean about how LLM's reasoned about LLMs.

I thought I would try evaluating a few different models on how well they could play tic-tac-toe and whether they played optimally. I also wanted to understand how much the typical board representations affected their ability to play. Essentially, most representations of tic-tac-toe found on the internet would be designed to look something like this:

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

How well could an LLM recognise it? The symbols in the second board have a simple mapping to those of the first and any human could identify this and play an optimal game of tic-tac-toe on either board. Can LLMs abstract the board state from these unusual representations?

You can see the code and analysis <a href="https://github.com/gdubbs100/LLMs_and_tictactoe">here</a>.

## Method
So I set up a simple tic-tac-toe environment with a textual state space that looked like the first state above. I set up some baseline algorithms - a random agent that plays a random move and an optimal agent which plays according to the <a href="https://en.wikipedia.org/wiki/Minimax">minmax algorithm</a>. 

I then selected some LLMs based on what I could access for free - either through cloud or to run locally on my laptop. In the end I mainly used <a href = "https://ollama.com/library/gemma4">Gemma4</a> with 31 billion parameters, but I also experimented with <a href="https://ollama.com/library/gpt-oss">GPT-OSS</a> with 120 billion parameters (an open source GPT model from OpenAI), and <a href = "https://huggingface.co/Qwen/Qwen3-0.6B">Qwen 3</a> with 0.6 billion parameters. I ran Gemma4 and GPT-OSS via Ollama cloud and Qwen 3 locally via HuggingFace.

I simulated games of tic-tac-toe and measured what proportion of moves were optimal (as per the minmax algorithm). Thanks to Claude, I was also able to build an interactive dashboard where I could review the results of games and the thinking traces of models. This was very useful initially as I was able to debug errors / failures of the LLM and also get an insight into what was actually happening in the reasoning process.

I started by running a simple tournament on a standard 3x3 board amongst random, minmax (hereafter referred to as `alphabeta`), Gemma4, GPT-OSS and Qwen to evaluate how well the language models performed on standard tic-tac-toe.
I found generally that large enough LLMs performed very well, so I started to play with the board representation to see how well they could reason about unusual symbols and representations of the board. I had limited resources as Ollama only offers so many free calls to cloud models per session. As a result, I often ended up focussing on the results for Gemma 4, as this model performed the best overall.

## Prompt design
I went through a few iterations of prompts. My intention was to design a prompt that would give enough information to the LLM so that it could play the game, but I wanted it to have to figure out _how_ to play. For example, I didn't want to tell it how to read the board, I wanted it to be able to look at the state and figure out what each of the symbols meant. I also didn't want to tell it that it was playing tic-tac-toe, as that comes with a whole bunch of knowledge it could query from its training data - instead, I just told it the rules. In most cases, the LLM figured out it was tic-tac-toe and mentioned as such in its reasoning traces anyway.

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

At each step I gave the agent the state of the board and information about what piece they were playing and their opponent was playing, as shown below:

```
You are playing as "O". Your opponent is "X".

State:
    |     |    
---------------
    |  X  |    
---------------
    |     |    
```

The decision to include information about the pieces was made because:
1. People know what pieces they are playing when they play the game
2. In initial experiments when I didn't give this information, the LLM spent tokens trying to figure out which piece it was using - which isn't really fair given 1.
3. If you don't assume that X moves first, board states with even numbers of pieces are actually ambiguous as to which player's turn it is

So I thought it was fair to include the information about pieces. However, even without this information, the LLMs were able to figure out which piece they were playing by assuming 3.

## Experiments

### Experiment 1: 3x3 Tic-Tac-Toe Tournament
In the first experiment I ran a tournament with language agents Qwen3 0.6B, GPT-OSS, Gemma4 and baselines random/alphabeta agents played 3x3 tic-tac-toe. Each agent played 10 games against every other agent and whether or not they won, lost, drew or lost due to an invalid move was recorded. Overall, the larger language agents (GPT-OSS and Gemma4) were able to play optimally, while Qwen3 struggled to even pick valid actions, resulting in a 100% loss rate. Overall results are below.

![Tournament Outcomes](/images/20260617tictactoe/3x3_tournament_outcome_rates.png)

As noted, the Qwen model was significantly smaller than the other LLMs used. The advantage of this was that it could run locally on my laptop (albiet slowly). But when thinking Qwen generated long, rambling traces that failed to achieve better answers. Without thinking, it would always select the action 0, regardless of whether it was occupied or not. To fix this, I tried using a fallback method, that re-prompted the LLM - stating the error (e.g. invalid action chosen, or a string rather than single integer was provided) and asked for a correction. Even then, Qwen appeared to be incredibly dense. For example, when the prompt stated: "Respond with a single integer only, no explanation.", it would respond with things like "The Answer is 1". Often, it would just choose 0, which was often occupied. As such, Qwen performed worse than the random agent, often loosing just because it could not correctly play a game.

GPT-OSS and Gemma4 performed optimally, or pretty close. Generally, their resoning traces showed that they recognised the game as tic-tac-toe and then started to evaluate possible strategies in this light. A very simple example of this is shown below in an adhoc dashboard I was able to whip up using Claude (very useful for debugging and evaluating LLM traces).

![Gemma4 recognises Tic-Tac-Toe](/images/20260617tictactoe/3x3_tournament_gemma4_vs_gptoss_trace.png)

This didn't mean they _always_ played optimally. For example, GPT-OSS missed a few optimal moves and as a result performed slightly worse than Gemma4. It lost one game to Gemma4 for no real reason. The board state is pictured below, with the highlighted yellow square indicated GPT-OSS chose to place an "O" on a cell that was already occupied with an "O". 

![GPT-OSS loss](/images/20260617tictactoe/3x3_tournament_gpt_oss_lossv2.png)

Basically GPT-OSS played a sub-optimal move (placed on O at position 1 instead of 2 or 6) on its previous attempt, which allowed Gemma4 to wedge it on this turn. However, rather than simply picking to block _one_ of the possible wins, it selected an invalid move. At first I thought this was a silly thing to do. However, on reflection - if you consider an optimal move, all the available (valid or invalid) moves result in a loss, so perhaps GPT-OSS still played optimally in this circumstance. But the prior move was sub-optimal, and I think this highlights the stochastic nature and unreliability of LLMs. Most of the time, they play optimally, except for the times when they don't.

My takeaways from this experiment was that larger LLMs are very capable at Tic-Tac-Toe, but I suspect that it may just be to significant exposure from their training data. That said, it is worth noting that they were able to competantly match the state representation with an integer selection, which is a type of abstract reasoning. Also, recognising and looking up the strategy for tic-tac-toe without being prompted is also pretty cool - thats a kind of pattern recognition to select a course of action. So despite tic-tac-toe being an easy game to play, I'm kinda impressed.

### Experiment 2: Changing the Board Representation
In the next batch of experiments, I tried to deviate from typical representations of tic-tac-toe to test how heavily LLMs were relying on the typical represations. The idea being that you can represent a 3x3 grid in text using a variety symbols, and if there is a simple pattern (like a 1:1 mapping) between a standard and non-standard representation, it should be easy enough for something intelligent to understand this. I used Gemma4 for these experiments because of its good performance on the tournament.

#### Initial tests
I started experimenting with a few simple changes e.g. replaceing "|" with ")" or an empty space with "-" to see how well Gemma could respond to these changes. It didn't really have any impact. Gemma would consistently achieve optimal scores and be able to draw with the optimal Alphabeta agent. So this is +1 in the LLMs' favour - even with representation adjustments, Gemma4 could play tic-tac-toe well. As a result, I tried finding board representations that would be deliberately confusing.

#### A more devious representation
I became a bit more devious in my choice of symbols so that they were deliberately confusing, but still readable and playable by a human (if perhaps painful and annoying). I eventually came up with the below design, where I replaced vertical ("|") and horizontal ("_") dividers with "X", and blank spaces with "O". I also replaced the pieces with "x" and "o".

```
 O  X  O  X  O 
XXXXXXXXXXXXXXX
 O  X  O  X  O 
XXXXXXXXXXXXXXX
 O  X  O  X  O 

```

For a human playing on this board, it would be a pain to parse, but not impossible, particularly if you know what each of the pieces represent. If you didn't, you might do a double take, but I don't think anyone couldn't figure it out after a bit. In some ways, I wonder if this might be easier for a computer to handle, because you are simply replacing one token with another, the visual factor which might be confusing for a human, may not be for an LLM or other program.

#### The Experiments
I ran three sub-experiments here, using the devious board representation. I varied the amount of information contained in the initial prompt for each:
1. In the first, I simply used the same initial prompt as used in the first experiment. It contained no information about the representation of the board except that it was a 3x3 grid.
2. In the second, I replaced the horizontal and vertical separaters of the board in the initial prompt with "X", representing what would be shown in the game state.
3. In the third, I used the same board representation in the initial prompt as 2., but also added a line stating "Empty cells are shown as the letter O." 

In each of these sub-experiments I played 30 games of alphabeta against Gemma4. I also benchmarked the games against 30 games of under the standard representation. Note that by playing the optimal alphabeta agent, the best possible outcome is a draw, which can only be achieved by playing optimally.

#### Results
Gemma4 struggled to play optimally under experiments 1 and 2, but managed to play optimally consistently in experiment 3. 

Below we show the outcome for Gemma 4 by the number of moves per game by experiment. Experiments 1 and 2 mostly resulted in losses due to invalid moves (i.e. moving to already occupied cells). Experiment 2 saw most invalid moves occurring later, perhaps indicating better performance than experiment 1.

![Gemma4 results](/images/20260617tictactoe/gemma4_moves_by_representation.png)

Across all of the experiment variants, Gemma 4 spent larger amounts of tokens thinking about these tasks relative to the standard baseline. Although, there was a notable decline in tokens per move for experiment 3. Gemma 4 has a context window of 256k tokens so there should not be any performance issues due to this filling up (see <a href="https://ai.google.dev/gemma/docs/core/model_card_4">here</a>).

![Token Useage](/images/20260617tictactoe/gemma4_tokens_per_move_by_representation.png)

Overall, Gemma4 often played optimally under all experiments, but often made invalid moves resulting in losses in experiments 1 and 2. These seem to result from challenges in parsing the board representation.

##### **Experiment 1**
Gemma struggled to play this variant (despite being successful at many others). It was only able to achieve a draw on a single match, and made invalid moves on 27/30 matches. It spent a significant amount of tokens trying to interpret the board state. As I intended with the design, it confused the "X" and "O" characters with actual pieces on the board, resulting in some entertaining reasoning, for example:

```
Let's try this:
    The board is actually:
    O X O
    X _ X
    O X O
    And the X's are already there.
    The only empty cell is 4.
    X moves at 4 and wins.
Response
4
```

This example was the initial move for the game, and so was actually optimal (4 is the centre of the board), but its reasoning is quite odd. Note that as part of the state prompt, shown each turn, we told it that it was playing "x" rather than "X". It didn't seem to register the difference between lower and upper cases. It also just hallucinated a board state as it wasn't able to parse the different symbols. It eventually lost this game by selecting an invalid action with a single square left open. 

![Gemma4 loss](/images/20260617tictactoe/gemma4_loss.png)

In the game above Gemma went first. This might be a bit more challenging as it may be ambiguous as to whether the upper/lower case was important. But when Alphabeta goes first, the inference is much simpler because both lower and upper case characters are present. According to the thinking traces, Gemma still struggled with this. In another game where Alphabeta had played "x" in the centre in the first move, Gemma played an invalid move in the first move, with the following logic:

```
If the board was already:
O X O
X O X
O X O
then the game would be over.
Since I'm asked for a move, cell 4 must be empty.

One last check. Could the board be:
O X O
X X X
O X O
And cell 4 is the 'x' which is empty?
Yes, that's what I've been assuming.
```

The reasoning trace doesn't seem to be coherent. Interestingly, its still treating the game as a game of tic-tac-toe and looking for the right cells. It just can't interpret the board.

##### **Experiment 2**
In this experiment there was not a major difference in performance from Experiment 1. Overall, Gemma4 struggled with the board representation still, even though we provided it with almost all the information it needed. Interestingly, it was able to identify "X" was the divider (provided in the prompt), but still got confused with whether "O" and "o" were different characters. For example:

```    
So the "X"s are dividers.
    The actual state is:
    Cell 0: O
    Cell 1: O
    Cell 2: O
    Cell 3: O
    Cell 4: x (lowercase x?)
    Cell 5: O
    Cell 6: O
    Cell 7: O
    Cell 8: O

Wait, this board is already full. 
```
After a significant amount of thinking, the model eventually concluded:

```
Let's think. If the user intended the 'O's to be the marks for 'o', then 'o' has already won. But it's the user's turn to move, so the game cannot be over.

Therefore, the 'O's must be empty cells.
```
So Gemma4 was able to nut the problem out. But, this isn't that hard a problem to figure out, as the per-turn prompt tells it that the opposition's piece is "o" rather than "O". I suspect that the reason Gemma4 wasn't very efficient at figuring this out may be due to its association of tic-tac-toe playing pieces with "X" and "O". Did this lead it to ignore information that was provided in the prompt in favour of its pre-existing bias?

Overall, Gemma4 would play mostly rationally, but would make silly mistakes and often choose already occupied cells. Much like its behaviour in experiment 1.

##### **Experiment 3**
In this experiment Gemma4 played 100% optimally, achieving draws on every game - much like it did on the standard games. This is pretty remarkable. The only difference between this experiment and the previous was the inclusion of a single line:
"Empty cells are shown as the letter O." 

The inclusion of this line drastically increases performance and reduces token usage. This surprises me as the conclusion that "O" represents empty cells seems to be easy to draw if you know:
1. pieces will be "x" or "o" (in prompt shown each turn)
2. "X" represents separators (in initial prompt)
3. There are 9 cells in the board (in intial prompt)
4. "o" != "O" (self-evident)

But based on the performance in the previous two experiments, it looks like this small fact created the most significant amount of confusion of any of the alterations in board representation. What I feel (but don't know how to demonstrate or prove) is that this is due to the LLM having a strong bias for "O" being a piece in tic-tac-toe, and it must devote a significant amount of effort (in tokens) to overcome this.

That said - in this experiment, Gemma4 still used around twice as many tokens as under the standard game. Oddly, it still devoted a lot of internal debate to whether or not the board was full or not. For example:

```

    State:
     O  X  O  X  O
    XXXXXXXXXXXXXXX
     O  X  O  X  O
    XXXXXXXXXXXXXXX
     O  X  O  X  O

    Looking at the provided layout:
     0  X  1  X  2
    XXXXXXXXXXXXXXX
     3  X  4  X  5
    XXXXXXXXXXXXXXX
     6  X  7  X  8

    The state is:
    Row 0: O (0), X (1), O (2)
    Row 1: O (3), X (4), O (5)
    Row 2: O (6), X (7), O (8)

    Wait, cell indices are:
    0 1 2
    3 4 5
    6 7 8

    X's positions: 1, 4, 7.
    O's positions: 0, 2, 3, 5, 6, 8.

    X has already won (1, 4, 7 is a column).

```

It is quite similar to what we observed in previous experiments in terms of thinking traces. But the agent just seems to be able to resolve those issues well enough to select valid actions and play tic-tac-toe competently.


## Conclusion
So it seems larger LLMs are pretty good at Tic-Tac-Toe. They play pretty close to optimally and are pretty good at untangling board states. But they _can_ be fooled (or at least Gemma4 can) by unusual representations. My key takeaway from these experiments has been that LLMs come with biases generated in training that take some effort to overcome. 

This exploration has left me intrigued by LLM's thinking traces. As the name suggests, they appear to show a process of thinking, but I'm not sure if that is really what it is. Often the process was repetitive nonsense generation or babble, with an occasional coherent revelation. Sometimes I saw completely incoherent babble lead to correct answers. I'm not sure what is going on here and I'd like to look at it in more detail. Perhaps it is some sort of monte-carlo sampling where the model refines the solution space? I'm sure there are papers on it somewhere and I should look them up.

I think it would be interesting to explore more complex problems, like connect 4, chess and so on. Or even create a new game which the LLM could not have been exposed to in the training data, requiring it to figure out a whole new approach. Another interesting avenue would be to see how well it could learn a new game via feedback - iteratively refining some prompt or knowledge base based on its previous performance.

### Post-script on Claude Code
I used Claude-code to do a lot of the coding for this project. Initially I designed some agent classes and then got Claude to build the rest. I didn't have much trouble using it as I knew what I wanted. Some useful things I did were getting Claude to build an ad-hoc dashboard for me to evaluate LLM traces / replay games and creating my first skill to quickly write experiment config files. 

Regarding the dashboard - I find data vis painful, especially dashboards (I did it for work for ~1.5 years). Being able to get Claude to write it made the process so simple. All I had to do was design it and then tell Claude to do it. There wasn't any tedious fiddling with switches and reactive elements and all that stuff - I could just ask for what I wanted and get it. I think because the objective of the task was to create something that I could verify visually, it was simple to create. I didn't care about the inner workings. In addition, creating this dashboard was a significant help with my project - kind of like how tensorboard can be for training neural networks. It helped me understand what the LLMs were thinking when, and see how they were playing the game. It was great!

The skill was also pretty cool. I found myself copying and pasting numerous configs, or asking Claude to do so. It just because easier to chuck a simple prompt in a text file and provide a schema for expected values. Every time I wanted a new config i'd just prompt Claude with `/create-experiment-config` and give a few extra instructions and it did it! It was very useful, you can see it <a href="https://github.com/gdubbs100/LLMs_and_tictactoe/tree/main/.claude/skills/create-experiment-config">here</a>
