#%%

def get_percentage(topics_list = list[str]) -> dict[str, float]:
    """
    Get the percentage of each topic in a list

    Parameters
    ----------
    topics_list : list[str]
        A list of topics

    Returns
    -------
    dict[str, float]
        A dictionary with the percentage of each topic
    """

    # Get the total number of topics
    total = len(topics_list)

    # Get the unique topics
    unique_N = list(set(topics_list))

    # Get the number of each topic
    N_count = {topic: topics_list.count(topic) for topic in unique_N}

    # Get the percentage of each topic
    N_percentage = {N: N_count[N] / total for N in unique_N}

    return N_percentage


#%% test the topic extraction code

if __name__ == "__main__":
    
    topics = ["Cars", "Cars", "Cars", "Cars", "Cars",
              "Toys", "Toys", "Toys", "Toys",
              "Dogs", "Dogs", "Dogs",
                "Cats", "Cats",
    ]
    
    print(get_percentage(topics))


# %%
