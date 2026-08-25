import json
from anthropic import Anthropic
from dotenv import load_dotenv
from app.config import AGENT_MODEL, AGENT_MAX_TOKENS
from agent.agent_tools import TOOLS, TOOLS_FUNCTIONS


# The Anthropic API key is needed to call the agent.
load_dotenv()


def ask_agent(user_input: str, system_prompt: str) -> dict:
    """
        Loop for the agent's investigation, which is a back and forth using Anthropic API.

        A question is sent with available tools (Python functions in this case), and a system prompt
        holding the instructions that apply to every round rather than to that one question.
        The model answers with suggestions to run some tools (in the form or arguments for a function).
        The tools are being run, the result is sent back as a new API request. Anthropic analyses the request and
        sends back another tool to be run, until the agent has enough information to answer the question.

        The conversation is held in a message that has to have all the history of the conversation, i.e. the tools
        calls, but also their id, their result, and so on from the beginning of the conversation.

        This means that sending 20k tokens in round X of a loop results in this 20k tokens being sent in every
        single round after round X. The number of token sent can only increase and accumulate round after round.

        Written after Claude's documentation
        https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview

        And specifically:
        https://platform.claude.com/docs/en/api/messages to format the message sent to Anthropic (class Message)
        https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use for tool calls
        https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls format for the tools results
        https://platform.claude.com/docs/en/build-with-claude/prompt-caching cost management: caching
    """

    # Create a client, reading ANTHROPIC_API_KEY from the .env file.
    client = Anthropic()

    # The API must have the whole conversation in each call.
    # All answers and calls are appended to the "messages" variable.
    # This means the longer the loop, the more expensive the investigation becomes.
    # Instantiating the messages variable:
    messages = [{"role": "user", "content": user_input}]

    # Counting the tokens being used per call for logs / cost optimization
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_written = 0
    total_cache_read = 0
    round_number = 0

    # Loop until the agent stops asking for tools. Each iteration runs the tools it asked for, appends their
    # results to the conversation, and asks it to continue.
    while True:
        round_number += 1

        # Call to Anthropic API. Asking Anthropic to cache the conversation.
        # Doc: https://platform.claude.com/docs/en/api/python/beta/messages/create
        response = client.messages.create(
            model=AGENT_MODEL,
            max_tokens=AGENT_MAX_TOKENS,
            cache_control={"type": "ephemeral"},
            system=system_prompt,
            tools=TOOLS,
            messages=messages
        )

        # Collecting tool_use blocks of the answer. Each one has to be run and the results
        # sent in a new request to the API.
        # Doc: https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use
        tool_uses = [block for block in response.content if block.type == "tool_use"]

        # Token usage can be extracted from the response for cost management and efficiency of the model.
        # Tokens for caching are separate and also need to be taken into account.
        # They need to be extracted and aggregated each round as the response does not hold cumulative token.
        uncached_input_tokens = response.usage.input_tokens

        # Cache tokens start to hit from a specific usage (depending on model). The key might be
        # missing from the response.
        cache_written = response.usage.cache_creation_input_tokens or 0
        cache_read = response.usage.cache_read_input_tokens or 0
        tokens_sent = uncached_input_tokens + cache_written + cache_read

        total_input_tokens += uncached_input_tokens
        total_output_tokens += response.usage.output_tokens
        total_cache_written += cache_written
        total_cache_read += cache_read

        print(f"\nRound {round_number}: {tokens_sent} tokens sent "
              f"({cache_read} read from the cache, {cache_written} written to it), "
              f"{response.usage.output_tokens} written back "
              f"(total so far: {total_input_tokens + total_cache_read + total_cache_written} in / {total_output_tokens} out). "
              f"{len(tool_uses)} tool calls")

        # The response content must be added to the conversation ("messages") from now on.
        # Doc: https://platform.claude.com/docs/en/agents-and-tools/tool-use/build-a-tool-using-agent#ring-2-the-agentic-loop
        messages.append({"role": response.role, "content": response.content})

        # The loop continues as long as the agent keeps calling for tools to be used,
        # which translates to response.stop_reason == "tool_use".
        # Any other stop_reason ends the loop.
        # Doc: https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons
        if response.stop_reason != "tool_use":

            # Checking if the AGENT_MAX_TOKENS params has been reached
            if response.stop_reason == "max_tokens":
                print(f"\tThe answer reached max_tokens ({AGENT_MAX_TOKENS}), the loop stops here")
                break

            # Handling other cases than "end_turn", which is the natural end of the investigation
            if response.stop_reason != "end_turn":
                print(f"\tThe investigation ended on {response.stop_reason} rather than end_turn")
            break

        # The tool the agent suggests to call have an id and an input (argument for the tools function).
        # The function's output will need to be added to the conversation and sent as new request.
        tool_results = []

        for call in tool_uses:
            print(f"\tThe agent is calling {call.name} ({call.input})")

            # Binding a tool result with the call id from the agent
            tool_result = {"type": "tool_result", "tool_use_id": call.id}

            try:
                # Running the tool with suggested input / argument from the agent.
                # TOOLS_FUNCTIONS references tools names with Python functions
                result = TOOLS_FUNCTIONS[call.name](**call.input)

                # Binding the result of the tool (function) to the response that will be sent back to the API
                tool_result["content"] = json.dumps(result, default=str)

            except Exception as e:
                tool_result["content"] = f"Error executing tool: {str(e)}"
                tool_result["is_error"] = True

            tool_results.append(tool_result)

        # All tool calls are being sent back in the next message
        messages.append({"role": "user", "content": tool_results})

    print(f"\nInvestigation over in {round_number} rounds, "
          f"{total_input_tokens + total_cache_read + total_cache_written} tokens sent "
          f"and {total_output_tokens} written back. "
          f"{total_cache_read} tokens were read from the cache and {total_cache_written} written to it\n")

    # Returning the overall response from the conversation + all tokens to calculate the cost of the request's loop

    result = {
        "response": response,
        "usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cache_read": total_cache_read,
            "total_cache_written": total_cache_written,
        }
    }

    return result


if __name__ == "__main__":

    from agent.agent_run import prompt_agent, system_prompt_agent
    response = ask_agent(user_input=prompt_agent, system_prompt=system_prompt_agent)


