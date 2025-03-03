def invoke_our_graph(user_input, callables, thread_id):
    # Ensure the callables parameter is a list as you can have multiple callbacks
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    # Invoke the graph with the current messages and callback configuration
    return (
        graph.invoke(
            {
                "user_request": user_input,
                "message_history": st.session_state["messages"],
            },
            config={"callbacks": callables, "configurable": {"thread_id": thread_id}},
        ),
        graph,
    )

# Handle new user input
user_message = st.chat_input("Message ChatCapitalHumain...")
if user_message:
    # Display and record the user's message
    st.chat_message("user", avatar=USER_AVATAR_PATH).write(user_message)
    st.session_state["messages"].append(HumanMessage(content=user_message))

    # Generate assistant response via your graph
    with st.chat_message("assistant", avatar=APP_ICON_PATH):
        response_placeholder = st.empty()
        streamlit_callback = get_streamlit_cb(st.empty())
        graph_response, graph = invoke_our_graph(
            user_message, [streamlit_callback], st.session_state["thread_id"]
        )
        st.info(graph)
        state = graph.get_state(
            {"configurable": {"thread_id": st.session_state["thread_id"]}}
        )
        st.warning(state)


        # final_response = graph_response["messages"][-1].content
        # st.session_state["messages"].append(AIMessage(content=final_response))
        # response_placeholder.write(final_response)

        # With multi_agents

        state = graph.get_state(
            {"configurable": {"thread_id": st.session_state["thread_id"]}}
        )
        print("state", state)
        #st.warning(graph_response)
        #st.info(graph_response["analysis_result"].response)
        st.session_state["messages"].append(
            AIMessage(content=graph_response["analysis_result"].response)
        )
        response_placeholder.write(graph_response)
        if graph_response.get("final_answer"):
            st.session_state["messages"].append(
                AIMessage(content=graph_response["final_answer"])
            )
            response_placeholder.write(graph_response["final_answer"])

        # TODO check state of the graph_response, if feedback in its name :

        # # if graph_response["analysis_result"].is_answerable == True:
        # further_feedack = st.text_input(
        #     "Veuillez donner plus de détails pour une meilleure réponse"
        # )
        # # if further_feedack:
        # print("further feedback", further_feedack)
        # graph.update_state(
        #     {"configurable": {"thread_id": st.session_state["thread_id"]}},
        #     {
        #         "human_analyst_feedback": "inclut aussi comme bons resultats tout ayant a partir de plus de 55%"
        #     },
        #     as_node="human_feedback",
        # )

        # for event in graph.stream(
        #     None,
        #     {"configurable": {"thread_id": st.session_state["thread_id"]}},
        #     stream_mode="updates",
        # ):
        #     print("--Node--")
        #     node_name = next(iter(event.keys()))
        #     print(node_name)
        # final_state = graph.get_state(
        #     {"configurable": {"thread_id": st.session_state["thread_id"]}}
        # )
        # print(final_state.values.get("final_query_instructions"))

        # st.session_state["messages"].append(AIMessage(content=str(graph_response)))

    # Save the conversation only after the assistant has responded,
    # and only if the user is logged in (i.e. email exists)
    # if st.experimental_user.get("email"):
    # save_chat_logs()
# Example usage
# config = {"configurable": {"thread_id": "123"}}
# result = graph.invoke({"user_request": "What is the total number of male students who answered 'Yes' to Question 3 in 2018?"}, config)


# # Input
# #user_request = "Nombre d'etudiant qui ont de bons resultats scolaires et vont sur pieds a l'ecole"
# user_request = "Nombre d'etudiant qui ont de bons resultats scolaires en 2018"
# #user_request = "Bonjour"

# thread = {"configurable": {"thread_id": "1"}}

# # Run the graph until the first interruption
# for event in graph.stream({"user_request":user_request}, thread, stream_mode="values"):
#     print(event)

# state = graph.get_state(thread)
# print(state.next)

# # If we are satisfied, then we simply supply no feedback
# further_feedack = "seems correct"
# graph.update_state(thread, {"human_analyst_feedback":
#                             further_feedack}, as_node="human_feedback")

# for event in graph.stream(None, thread, stream_mode="updates"):
#     print("--Node--")
#     node_name = next(iter(event.keys()))
#     print(node_name)


# final_state = graph.get_state(thread)
# print(final_state.values.get('final_query_instructions'))
