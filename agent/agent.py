import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from tools.data_tools import agent_instance


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# =========================================================
# LLM INSTRUCTIONS
# =========================================================

AGENT_INSTRUCTIONS = """
You are a data analysis agent for fuel transaction data.

Your job is to answer the user's questions accurately by using the
available analytical tools.

Important rules:

1. Never invent values, companies, card numbers, vehicles, quantities,
   prices, products, or analytical results.

2. Use tools whenever the answer depends on the dataset.

3. You may call multiple tools sequentially when necessary.

4. Prefer analytical tools such as:
   - find_extreme
   - group_and_aggregate
   - group_by_multiple
   - calculate_sum
   - calculate_mean
   - calculate_statistics

   instead of retrieving individual transaction records.

5. Use search_transactions only when the user explicitly needs
   transaction-level records.

6. If you do not know the exact dataset column names, call get_columns.

7. If you do not know the exact categorical values in a column,
   call get_column_values.

8. Fuel consumption / quantity normally refers to the transaction
   quantity column. Determine the correct column from the dataset.

9. When the user says "diesel" or "diesel fuel", include all diesel
   products unless they explicitly specify one particular diesel product.

10. When comparing companies, use the Company column.

11. When comparing fuel cards, use the Number column.

12. A question can contain multiple analytical requests.
    Solve every part before giving the final answer.

13. Do not calculate large aggregations yourself from raw transaction
    records. Use the aggregation tools so Python/Pandas performs the
    calculation.

14. When the user asks for "most", "highest", "largest", "least",
    "lowest", etc., prefer find_extreme.

15. Answer the user in Bulgarian unless they ask for another language.

16. Present numerical results clearly and mention the metric that was
    used.

17. If a tool returns an error because a column or value was incorrect,
    inspect the available columns or values and try again.

18. Do not stop after the first tool call if more information is required
    to completely answer the user's question.
"""


# =========================================================
# TOOL DEFINITIONS
# =========================================================

tools = [

    # -----------------------------------------------------
    # DATASET INFORMATION
    # -----------------------------------------------------

    {
        "type": "function",
        "name": "get_dataset_shape",
        "description": (
            "Returns the number of rows and columns in the enriched "
            "fuel transactions dataset."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },

    {
        "type": "function",
        "name": "get_columns",
        "description": (
            "Returns all available dataset columns together with their data types. "
            "Use this tool when the required column names are not known."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },

    {
        "type": "function",
        "name": "get_column_values",
        "description": (
            "Returns the most common unique values from a dataset column. "
            "Use this to discover exact product names, company names, "
            "card numbers, stations, vehicles, or other categorical values."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "Exact dataset column name."
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of values to return. "
                        "Use a small value unless more values are necessary."
                    )
                }
            },
            "required": ["column"]
        }
    },

    # -----------------------------------------------------
    # BASIC STATISTICS
    # -----------------------------------------------------

    {
        "type": "function",
        "name": "calculate_mean",
        "description": (
            "Calculates the arithmetic mean of a numeric dataset column. "
            "Optional filters can restrict which transactions are included."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "Exact numeric dataset column to average."
                },
                "filters": {
                    "type": "array",
                    "description": "Optional transaction filters.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {
                                "type": "string"
                            },
                            "operator": {
                                "type": "string",
                                "enum": [
                                    "eq",
                                    "neq",
                                    "contains",
                                    "not_contains",
                                    "gt",
                                    "gte",
                                    "lt",
                                    "lte",
                                    "in"
                                ]
                            },
                            "value": {}
                        },
                        "required": [
                            "column",
                            "operator",
                            "value"
                        ]
                    }
                }
            },
            "required": ["column"]
        }
    },

    {
        "type": "function",
        "name": "calculate_sum",
        "description": (
            "Calculates the total sum of a numeric dataset column. "
            "Use this for total fuel quantity, total transaction amount, "
            "total cost, or other totals. Filters may be applied."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "Exact numeric dataset column to sum."
                },
                "filters": {
                    "type": "array",
                    "description": "Optional transaction filters.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {
                                "type": "string"
                            },
                            "operator": {
                                "type": "string",
                                "enum": [
                                    "eq",
                                    "neq",
                                    "contains",
                                    "not_contains",
                                    "gt",
                                    "gte",
                                    "lt",
                                    "lte",
                                    "in"
                                ]
                            },
                            "value": {}
                        },
                        "required": [
                            "column",
                            "operator",
                            "value"
                        ]
                    }
                }
            },
            "required": ["column"]
        }
    },

    {
        "type": "function",
        "name": "calculate_statistics",
        "description": (
            "Calculates descriptive statistics for a numeric column: "
            "count, sum, mean, median, minimum, maximum, and standard deviation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "Exact numeric dataset column."
                },
                "filters": {
                    "type": "array",
                    "description": "Optional transaction filters.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {
                                "type": "string"
                            },
                            "operator": {
                                "type": "string",
                                "enum": [
                                    "eq",
                                    "neq",
                                    "contains",
                                    "not_contains",
                                    "gt",
                                    "gte",
                                    "lt",
                                    "lte",
                                    "in"
                                ]
                            },
                            "value": {}
                        },
                        "required": [
                            "column",
                            "operator",
                            "value"
                        ]
                    }
                }
            },
            "required": ["column"]
        }
    },

    # -----------------------------------------------------
    # GROUPING AND AGGREGATION
    # -----------------------------------------------------

    {
        "type": "function",
        "name": "group_and_aggregate",
        "description": (
            "Groups fuel transactions by one column and aggregates another column. "
            "Use this for rankings and comparisons between companies, cards, "
            "vehicles, products, stations, or other categories. "
            "For example: top 10 companies by total fuel consumption, "
            "total liters by card, average price by product."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "string",
                    "description": (
                        "Exact dataset column used to group transactions, "
                        "for example Company, Number, Material, or Name."
                    )
                },
                "value_column": {
                    "type": "string",
                    "description": (
                        "Exact numeric column to aggregate."
                    )
                },
                "aggregation": {
                    "type": "string",
                    "enum": [
                        "sum",
                        "mean",
                        "count",
                        "min",
                        "max",
                        "median"
                    ],
                    "description": "Aggregation operation."
                },
                "filters": {
                    "type": "array",
                    "description": (
                        "Optional filters applied before grouping."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {
                                "type": "string"
                            },
                            "operator": {
                                "type": "string",
                                "enum": [
                                    "eq",
                                    "neq",
                                    "contains",
                                    "not_contains",
                                    "gt",
                                    "gte",
                                    "lt",
                                    "lte",
                                    "in"
                                ]
                            },
                            "value": {}
                        },
                        "required": [
                            "column",
                            "operator",
                            "value"
                        ]
                    }
                },
                "sort": {
                    "type": "string",
                    "enum": [
                        "asc",
                        "desc"
                    ],
                    "description": (
                        "Sort aggregated results ascending or descending."
                    )
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of grouped results returned."
                    )
                }
            },
            "required": [
                "group_by",
                "value_column"
            ]
        }
    },

    {
        "type": "function",
        "name": "group_by_multiple",
        "description": (
            "Groups transactions using multiple columns and aggregates "
            "a numeric value. Use for analyses such as consumption by "
            "company and product, company and card, card and product, "
            "or vehicle and product."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": (
                        "List of exact columns to group by."
                    )
                },
                "value_column": {
                    "type": "string",
                    "description": (
                        "Exact numeric column to aggregate."
                    )
                },
                "aggregation": {
                    "type": "string",
                    "enum": [
                        "sum",
                        "mean",
                        "count",
                        "min",
                        "max",
                        "median"
                    ]
                },
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {
                                "type": "string"
                            },
                            "operator": {
                                "type": "string",
                                "enum": [
                                    "eq",
                                    "neq",
                                    "contains",
                                    "not_contains",
                                    "gt",
                                    "gte",
                                    "lt",
                                    "lte",
                                    "in"
                                ]
                            },
                            "value": {}
                        },
                        "required": [
                            "column",
                            "operator",
                            "value"
                        ]
                    }
                },
                "sort": {
                    "type": "string",
                    "enum": [
                        "asc",
                        "desc"
                    ]
                },
                "limit": {
                    "type": "integer"
                }
            },
            "required": [
                "group_by",
                "value_column"
            ]
        }
    },

    {
        "type": "function",
        "name": "find_extreme",
        "description": (
            "Finds the group with the highest or lowest aggregated value. "
            "This is the preferred tool when the user asks which company, "
            "card, vehicle, product, station, or other category has the most, "
            "least, highest, or lowest value."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "string",
                    "description": (
                        "Exact column identifying the groups to compare."
                    )
                },
                "value_column": {
                    "type": "string",
                    "description": (
                        "Exact numeric column to aggregate."
                    )
                },
                "aggregation": {
                    "type": "string",
                    "enum": [
                        "sum",
                        "mean",
                        "count",
                        "min",
                        "max",
                        "median"
                    ]
                },
                "extreme": {
                    "type": "string",
                    "enum": [
                        "max",
                        "min"
                    ],
                    "description": (
                        "Use max for the highest value and min for the lowest."
                    )
                },
                "filters": {
                    "type": "array",
                    "description": (
                        "Optional filters applied before comparison."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {
                                "type": "string"
                            },
                            "operator": {
                                "type": "string",
                                "enum": [
                                    "eq",
                                    "neq",
                                    "contains",
                                    "not_contains",
                                    "gt",
                                    "gte",
                                    "lt",
                                    "lte",
                                    "in"
                                ]
                            },
                            "value": {}
                        },
                        "required": [
                            "column",
                            "operator",
                            "value"
                        ]
                    }
                }
            },
            "required": [
                "group_by",
                "value_column"
            ]
        }
    },

    # -----------------------------------------------------
    # COUNTS
    # -----------------------------------------------------

    {
        "type": "function",
        "name": "value_counts",
        "description": (
            "Counts how often each value appears in a categorical column. "
            "Use for transaction counts by company, product, card, vehicle, "
            "station, or another category."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string"
                },
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {
                                "type": "string"
                            },
                            "operator": {
                                "type": "string",
                                "enum": [
                                    "eq",
                                    "neq",
                                    "contains",
                                    "not_contains",
                                    "gt",
                                    "gte",
                                    "lt",
                                    "lte",
                                    "in"
                                ]
                            },
                            "value": {}
                        },
                        "required": [
                            "column",
                            "operator",
                            "value"
                        ]
                    }
                },
                "limit": {
                    "type": "integer"
                }
            },
            "required": ["column"]
        }
    },

    {
        "type": "function",
        "name": "count_unique",
        "description": (
            "Counts unique values in a dataset column. "
            "Use for questions such as how many unique companies, cards, "
            "vehicles, products, or stations exist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string"
                },
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {
                                "type": "string"
                            },
                            "operator": {
                                "type": "string",
                                "enum": [
                                    "eq",
                                    "neq",
                                    "contains",
                                    "not_contains",
                                    "gt",
                                    "gte",
                                    "lt",
                                    "lte",
                                    "in"
                                ]
                            },
                            "value": {}
                        },
                        "required": [
                            "column",
                            "operator",
                            "value"
                        ]
                    }
                }
            },
            "required": ["column"]
        }
    },

    # -----------------------------------------------------
    # TRANSACTION SEARCH
    # -----------------------------------------------------

    {
        "type": "function",
        "name": "search_transactions",
        "description": (
            "Returns a limited number of individual transaction records "
            "matching specified filters. Use this when the user explicitly "
            "wants to inspect individual transactions. Do not use this tool "
            "for large aggregate calculations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {
                                "type": "string"
                            },
                            "operator": {
                                "type": "string",
                                "enum": [
                                    "eq",
                                    "neq",
                                    "contains",
                                    "not_contains",
                                    "gt",
                                    "gte",
                                    "lt",
                                    "lte",
                                    "in"
                                ]
                            },
                            "value": {}
                        },
                        "required": [
                            "column",
                            "operator",
                            "value"
                        ]
                    }
                },
                "columns": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": (
                        "Optional list of columns to return."
                    )
                },
                "sort_by": {
                    "type": "string"
                },
                "sort": {
                    "type": "string",
                    "enum": [
                        "asc",
                        "desc"
                    ]
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of transaction records returned."
                    )
                }
            },
            "required": []
        }
    },

    # -----------------------------------------------------
    # CARD INFORMATION
    # -----------------------------------------------------

    {
        "type": "function",
        "name": "get_card_info",
        "description": (
            "Returns information about a specific fuel card. "
            "Use this to find the company and vehicle associated with "
            "a specific card number."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "card_number": {
                    "type": "string",
                    "description": (
                        "Exact fuel card number."
                    )
                }
            },
            "required": [
                "card_number"
            ]
        }
    },

    {
        "type": "function",
        "name": "get_transactions_by_card",
        "description": (
            "Returns individual fuel transactions for one specific card. "
            "Use when the user explicitly asks to inspect transactions "
            "made with a particular fuel card."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "card_number": {
                    "type": "string",
                    "description": (
                        "Exact fuel card number."
                    )
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of transactions returned."
                    )
                }
            },
            "required": [
                "card_number"
            ]
        }
    }
]


# =========================================================
# USER QUESTION
# =========================================================

input_messages = [
    {
        "role": "user",
        "content": (
            "Искам да сметнеш общото количество на потреблението на фирма сийд експрес, но само за горивата"
        )
    }
]


# =========================================================
# AGENT LOOP
# =========================================================

MAX_TOOL_ROUNDS = 15
tool_round = 0


while True:

    tool_round += 1

    if tool_round > MAX_TOOL_ROUNDS:
        print(
            "\nAgent stopped because the maximum number "
            "of tool rounds was reached."
        )
        break

    response = client.responses.create(
        model="gpt-5-mini",
        instructions=AGENT_INSTRUCTIONS,
        input=input_messages,
        tools=tools,
        tool_choice="auto"
    )

    # Keep model output in conversation history
    input_messages.extend(response.output)

    # Find function calls requested by the model
    function_calls = [
        item
        for item in response.output
        if item.type == "function_call"
    ]

    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    if not function_calls:
        print("\nAssistant:")
        print(response.output_text)
        break

    # =====================================================
    # EXECUTE TOOLS
    # =====================================================

    for item in function_calls:

        print("\n" + "=" * 60)
        print("Tool requested:")
        print(item.name)

        print("\nArguments:")
        print(item.arguments)

        try:

            arguments = json.loads(
                item.arguments or "{}"
            )

            # -------------------------------------------------
            # DATASET INFORMATION
            # -------------------------------------------------

            if item.name == "get_dataset_shape":

                result = (
                    agent_instance
                    .get_dataset_shape()
                )

            elif item.name == "get_columns":

                result = (
                    agent_instance
                    .get_columns()
                )

            elif item.name == "get_column_values":

                result = (
                    agent_instance
                    .get_column_values(
                        column=arguments["column"],
                        limit=arguments.get(
                            "limit",
                            50
                        )
                    )
                )

            # -------------------------------------------------
            # BASIC STATISTICS
            # -------------------------------------------------

            elif item.name == "calculate_mean":

                result = (
                    agent_instance
                    .calculate_mean(
                        column=arguments["column"],
                        filters=arguments.get(
                            "filters"
                        )
                    )
                )

            elif item.name == "calculate_sum":

                result = (
                    agent_instance
                    .calculate_sum(
                        column=arguments["column"],
                        filters=arguments.get(
                            "filters"
                        )
                    )
                )

            elif item.name == "calculate_statistics":

                result = (
                    agent_instance
                    .calculate_statistics(
                        column=arguments["column"],
                        filters=arguments.get(
                            "filters"
                        )
                    )
                )

            # -------------------------------------------------
            # GROUPING / ANALYTICS
            # -------------------------------------------------

            elif item.name == "group_and_aggregate":

                result = (
                    agent_instance
                    .group_and_aggregate(
                        group_by=arguments[
                            "group_by"
                        ],
                        value_column=arguments[
                            "value_column"
                        ],
                        aggregation=arguments.get(
                            "aggregation",
                            "sum"
                        ),
                        filters=arguments.get(
                            "filters"
                        ),
                        sort=arguments.get(
                            "sort",
                            "desc"
                        ),
                        limit=arguments.get(
                            "limit",
                            20
                        )
                    )
                )

            elif item.name == "group_by_multiple":

                result = (
                    agent_instance
                    .group_by_multiple(
                        group_by=arguments[
                            "group_by"
                        ],
                        value_column=arguments[
                            "value_column"
                        ],
                        aggregation=arguments.get(
                            "aggregation",
                            "sum"
                        ),
                        filters=arguments.get(
                            "filters"
                        ),
                        sort=arguments.get(
                            "sort",
                            "desc"
                        ),
                        limit=arguments.get(
                            "limit",
                            30
                        )
                    )
                )

            elif item.name == "find_extreme":

                result = (
                    agent_instance
                    .find_extreme(
                        group_by=arguments[
                            "group_by"
                        ],
                        value_column=arguments[
                            "value_column"
                        ],
                        aggregation=arguments.get(
                            "aggregation",
                            "sum"
                        ),
                        extreme=arguments.get(
                            "extreme",
                            "max"
                        ),
                        filters=arguments.get(
                            "filters"
                        )
                    )
                )

            # -------------------------------------------------
            # COUNTS
            # -------------------------------------------------

            elif item.name == "value_counts":

                result = (
                    agent_instance
                    .value_counts(
                        column=arguments["column"],
                        filters=arguments.get(
                            "filters"
                        ),
                        limit=arguments.get(
                            "limit",
                            20
                        )
                    )
                )

            elif item.name == "count_unique":

                result = (
                    agent_instance
                    .count_unique(
                        column=arguments["column"],
                        filters=arguments.get(
                            "filters"
                        )
                    )
                )

            # -------------------------------------------------
            # TRANSACTIONS
            # -------------------------------------------------

            elif item.name == "search_transactions":

                result = (
                    agent_instance
                    .search_transactions(
                        filters=arguments.get(
                            "filters"
                        ),
                        columns=arguments.get(
                            "columns"
                        ),
                        sort_by=arguments.get(
                            "sort_by"
                        ),
                        sort=arguments.get(
                            "sort",
                            "desc"
                        ),
                        limit=arguments.get(
                            "limit",
                            50
                        )
                    )
                )

            # -------------------------------------------------
            # CARD INFORMATION
            # -------------------------------------------------

            elif item.name == "get_card_info":

                result = (
                    agent_instance
                    .get_card_info(
                        card_number=arguments[
                            "card_number"
                        ]
                    )
                )

            elif item.name == "get_transactions_by_card":

                result = (
                    agent_instance
                    .get_transactions_by_card(
                        card_number=arguments[
                            "card_number"
                        ],
                        limit=arguments.get(
                            "limit",
                            100
                        )
                    )
                )

            # -------------------------------------------------
            # UNKNOWN TOOL
            # -------------------------------------------------

            else:

                result = {
                    "error": (
                        f"Unknown tool: "
                        f"{item.name}"
                    )
                }

            # -------------------------------------------------
            # SERIALIZE RESULT
            # -------------------------------------------------

            output = json.dumps(
                result,
                ensure_ascii=False,
                default=str
            )

        except Exception as e:

            result = {
                "error": str(e),
                "tool": item.name
            }

            output = json.dumps(
                result,
                ensure_ascii=False,
                default=str
            )

        # -------------------------------------------------
        # DEBUG OUTPUT
        # -------------------------------------------------

        print("\nTool result:")
        print(result)

        # -------------------------------------------------
        # SEND TOOL RESULT BACK TO MODEL
        # -------------------------------------------------

        input_messages.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": output
            }
        )