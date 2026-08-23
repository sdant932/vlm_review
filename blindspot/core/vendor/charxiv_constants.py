"""Vendored verbatim from princeton-nlp/CharXiv `src/constants.py` (retrieved 2026-08-21).

Everything CharXiv's official protocol needs, so our numbers are comparable to theirs:

  DESCRIPTIVE_RESP_INST   the 19 descriptive question templates ({} takes the subplot prefix)
  DESCRIPTIVE_GRADING_QMAP short human-readable label per template id
  REASONING_RESP_INST     answer-format wrapper for reasoning questions, keyed by
                          `reasoning_a_type` (1-4) -- CharXiv does NOT send the bare question
  *_GRADING_PREFIX/ICL    the official LLM-judge prompt and its seven per-type rubrics
"""

DESCRIPTIVE_RESP_INST = {1: '{}what is its title?\n'
    '    * Your final answer should be the most relevant title of the plot that is explicitly '
    'written.\n'
    "    * If the plot does not have an explicit title or contains only a letter, answer 'Not "
    "Applicable'.\n"
    '    ',
 2: '{}what is the label of the x-axis?\n'
    '    * Your final answer should be the label of the x-axis that is explicitly written, '
    'including the case when x-axis is shared across multiple subplots. When the x-axis is present '
    'on both the top and bottom of the plot, answer the label of the x-axis at the bottom.\n'
    "    * If the plot does not have an explicit x-axis label, answer 'Not Applicable'.\n"
    '    ',
 3: '{}what is the label of the y-axis?\n'
    '    * Your final answer should be the label of the y-axis that is explicitly written, '
    'including the case when y-axis is shared across multiple subplots. When the y-axis is present '
    'on both the left and right of the plot, answer the label of the y-axis at the left.\n'
    "    * If the plot does not have an explicit y-axis label, answer 'Not Applicable'.",
 4: '{}what is the leftmost labeled tick on the x-axis?\n'
    '    * Your final answer should be the tick value on the x-axis that is explicitly written, '
    'including the case when x-axis is shared across multiple subplots. When the x-axis is present '
    'on both the top and bottom of the plot, answer based on the axis at the bottom. Ignore units '
    'or scales that are written separately from the tick, such as units and scales from the axis '
    'label or the corner of the plot.',
 5: '{}what is the rightmost labeled tick on the x-axis?\n'
    '    * Your final answer should be the tick value on the x-axis that is explicitly written, '
    'including the case when x-axis is shared across multiple subplots. When the x-axis is present '
    'on both the top and bottom of the plot, answer based on the axis at the bottom. Ignore units '
    'or scales that are written separately from the tick, such as units and scales from the axis '
    'label or the corner of the plot.',
 6: '{}what is the spatially lowest labeled tick on the y-axis?\n'
    '    * Your final answer should be the tick value on the y-axis that is explicitly written, '
    'including the case when y-axis is shared across multiple subplots. When the y-axis is present '
    'on both the left and right of the plot, based on the axis at the left. Ignore units or scales '
    'that are written separately from the tick, such as units and scales from the axis label or '
    'the corner of the plot.',
 7: '{}what is the spatially highest labeled tick on the y-axis?\n'
    '    * Your final answer should be the tick value on the y-axis that is explicitly written, '
    'including the case when y-axis is shared across multiple subplots. When the y-axis is present '
    'on both the left and right of the plot, based on the axis at the left. Ignore units or scales '
    'that are written separately from the tick, such as units and scales from the axis label or '
    'the corner of the plot.',
 8: '{}what is difference between consecutive numerical tick values on the x-axis?\n'
    '    * Your final answer should be the difference between consecutive numerical tick values of '
    'the x-axis, including the case when x-axis is shared across multiple subplots. When the '
    'x-axis is present on both the top and bottom of the plot, answer based on the axis at the '
    'bottom. Ignore units or scales that are written separately from the tick, such as units and '
    'scales from the axis label or the corner of the plot.\n'
    '    * If the plot does not have an explicit x-axis tick value, or if the tick values are not '
    'numerical, or if the difference is not constant between all consecutive tick values, answer '
    '"Not Applicable".',
 9: '{}what is difference between consecutive numerical tick values on the y-axis?\n'
    '    * Your final answer should be the difference between consecutive numerical tick values of '
    'the y-axis, including the case when y-axis is shared across multiple subplots. When the '
    'y-axis is present on both the left and right of the plot, answer based on the axis at the '
    'left. Ignore units or scales that are written separately from the tick, such as units and '
    'scales from the axis label or the corner of the plot.\n'
    '    * If the plot does not have an explicit y-axis tick value, or if the tick values are not '
    'numerical, or if the difference is not constant between all consecutive tick values, answer '
    '"Not Applicable".',
 10: '{}how many lines are there?\n'
     '    * Your final answer should be the number of lines in the plot. Ignore grid lines, tick '
     'marks, and any vertical or horizontal auxiliary lines.\n'
     '    * If the plot does not contain any lines or is not considered a line plot, answer "Not '
     'Applicable".',
 11: '{}do any lines intersect?\n'
     '    * Your final answer should be "Yes" if any lines intersect, and "No" otherwise. Ignore '
     'grid lines, tick marks, and any vertical or horizontal auxiliary lines.\n'
     '    * If the plot does not contain any lines or is not considered a line plot, answer "Not '
     'Applicable".',
 12: '{}how many discrete labels are there in the legend?\n'
     '    * Your final answer should account for only labels relevant to the plot in the legend, '
     'even if the legend is located outside the plot. \n'
     '    * If the plot does not have a legend or no legend is not considered relevant to this '
     'plot, answer "Not Applicable".',
 13: '{}what are the names of the labels in the legend?\n'
     '    * You should write down the labels from top to bottom, then from left to right and '
     'separate the labels with commas. Your final answer should account for only labels relevant '
     'to the plot in the legend, even if the legend is located outside the plot.\n'
     '    * If the plot does not have a legend or no legend is not considered relevant to this '
     'plot, answer "Not Applicable".',
 14: '{}what is the difference between the maximum and minimum values of the tick labels on the '
     'continuous legend (i.e., colorbar)?\n'
     '    * You should remove the percentage sign (if any) in your answer.\n'
     '    * If the plot does not have an explicit colorbar-based continuous legend or the legend '
     'is not considered relevant to this subplot, answer "Not Applicable".',
 15: '{}what is the maximum value of the tick labels on the continuous legend (i.e., colorbar)?\n'
     '    * You should remove the percentage sign (if any) in your answer. \n'
     '    * If the plot does not have an explicit colorbar-based continuous legend or the legend '
     'is not considered relevant to this subplot, answer "Not Applicable".',
 16: '{}what is the general trend of data from left to right?\n'
     '    * Your final answer should be within a few words, such as "increases", "increases then '
     'stabilizes".',
 17: '{}What is the total number of explicitly labeled ticks across all axes?\n'
     '    * Your final answer should be the total number of explicitly labeled ticks across all '
     'axes, including the case when any axis is shared across multiple subplots.',
 18: 'What is the layout of the subplots?\n'
     '    * Your final answer should follow "n by m" format, where n is the number of rows and m '
     'is the number of columns.\n'
     '    * If the plot does not contain subplots, answer "1 by 1".',
 19: 'What is the number of subplots?\n'
     '    * Your final answer should be the total number of subplots in the plot.\n'
     '    * If the plot does not contain subplots, answer "1".'}

DESCRIPTIVE_GRADING_QMAP = {1: 'What is the title of the plot?',
 2: 'What is the label of the x-axis?',
 3: 'What is the label of the y-axis?',
 4: 'What is the leftmost labeled tick on the x-axis?',
 5: 'What is the rightmost labeled tick on the x-axis?',
 6: 'What is the spatially lowest labeled tick on the y-axis?',
 7: 'What is the spatially highest labeled tick on the y-axis?',
 8: 'What is difference between consecutive numerical tick values on the x-axis?',
 9: 'What is difference between consecutive numerical tick values on the y-axis?',
 10: 'How many lines are there?',
 11: 'Do any lines intersect?',
 12: 'How many discrete labels are there in the legend?',
 13: 'What are the names of the labels in the legend? (from top to bottom, then left to right)',
 14: 'What is the difference between the maximum and minimum values of the tick labels on the '
     'continuous legend (i.e., colorbar)?',
 15: 'What is the maximum value of the tick labels on the continuous legend (i.e., colorbar)?',
 16: 'What is the general trend of data from left to right?',
 17: 'What is the total number of explicitly labeled ticks across all axes?',
 18: 'What is the layout of the subplots?',
 19: 'What is the number of subplots?'}

DESCRIPTIVE_GRADING_PREFIX = ('\n'
 'You will be given <|NUM_TRIPLETS|> pairs of ground truth answers and model responses under an '
 'overarching question. You need to go through each of the pairs, extract the final answer from '
 'the model response, compare it with the ground truth answer, and then assign a binary score. '
 'Avoid providing explanations in your response. If there is no provided model response, please '
 'leave the extracted answer empty and give a score of 0. Your response must follow json formats '
 'with keys [<|JSON_KEYS|>] where the value for any `extract_answer` is your extracted answer and '
 '`score` is an interger in [0, 1] based on the following rules:\n'
 '\n'
 '\n'
 'Overarching Question: <|OVERARCHING_QUESTION|>\n')

DESCRIPTIVE_GRADING_ICL = {'bool': '\n'
         'Rubric:\n'
         '    * Give a score of 1 if and only if the extracted answer and the ground truth answer '
         'are the same.\n'
         '    * Give a score of 0 if the extracted answer and the ground truth answer are '
         'different.\n'
         '    * When ground truth answer is "Not Applicable", the response must express "Not '
         'Applicable" to receive a score of 1.\n'
         '\n'
         '    ### Example Start ###\n'
         '    T1:\n'
         '    Response 1: No, there are no intersections.\n'
         '    Ground Truth 1: no\n'
         '\n'
         '    T2:\n'
         '    Response 2: No, all the lines are parallel.\n'
         '    Ground Truth 2: Yes\n'
         '\n'
         '    T3:\n'
         '    Response 3: There are no lines in the plot.\n'
         '    Ground Truth 3: Not Applicable\n'
         '\n'
         '    {\n'
         '        "extract_answer_T1": "No",\n'
         '        "score_T1": 1\n'
         '        "extract_answer_T2: "No",\n'
         '        "score_T2": 0\n'
         '        "extract_answer_T3": "Not Applicable",\n'
         '        "score_T3": 1\n'
         '    }\n'
         '    ### Example End ###   \n',
 'enum': '\n'
         'Rubric:\n'
         '    * Give a score of 1 if and only if the extracted answer and the ground truth answer '
         "are referring to the same term. It's acceptable to have equivalent grammar or form "
         '(e.g., α and alpha; $R^2_{t,h,v,m}$ and R^2_t,h,v,m). The order of the terms must be the '
         'same.\n'
         '    * Give a score of 0 if any term in the extracted answer is different from the ground '
         'truth answer, or if the order of the terms is different.\n'
         '    * When ground truth answer is "Not Applicable", the response must express "Not '
         'Applicable" to receive a score of 1.\n'
         '\n'
         '    ### Example Start ###\n'
         '    T1:\n'
         '    Response 1: Here are the names of the labels: A, B, C\n'
         '    Ground Truth 1: B, A, C\n'
         '\n'
         '    T2:\n'
         '    Response 2: The labels are T56, B33.\n'
         '    Ground Truth 2: T56,B33,A12\n'
         '\n'
         '    T3:\n'
         '    Response 3: \x07lpha, \x08eta, \\gamma^t_v\n'
         '    Ground Truth 3: α, β, γ_v^t\n'
         '\n'
         '    {\n'
         '        "extract_answer_T1": "A, B, C",\n'
         '        "score_T1": 0\n'
         '        "extract_answer_T2: "T56, B33",\n'
         '        "score_T2": 0\n'
         '        "extract_answer_T3": "\x07lpha, \x08eta, \\gamma^t_v",\n'
         '        "score_T3": 1\n'
         '    }\n'
         '    ### Example End ###\n',
 'layout': '\n'
           'Rubric:\n'
           '    * Give a score of 1 if and only if the extracted answer and the ground truth '
           'answer are the same in terms of the number of rows and columns (e.g., n by m).\n'
           '    * Give a score of 0 if the extracted answer is different from the ground truth '
           'answer.\n'
           '\n'
           '    ### Example Start ###\n'
           '    T1:\n'
           '    Response 1: 2 by 3\n'
           '    Ground Truth 1: 3 by 2\n'
           '\n'
           '    T2:\n'
           '    Response 2: the layout is 1 by 1\n'
           '    Ground Truth 2: 1 by 1\n'
           '\n'
           '    T3:\n'
           '    Response 3: there are two rows and three columns\n'
           '    Ground Truth 3: 2 by 3\n'
           '\n'
           '    {\n'
           '        "extract_answer_T1": "2 by 3",\n'
           '        "score_T1": 0\n'
           '        "extract_answer_T2: "1 by 1",\n'
           '        "score_T2": 1\n'
           '        "extract_answer_T3": "2 by 3",\n'
           '        "score_T3": 1\n'
           '    }\n'
           '    ### Example End ###\n',
 'ocr': '\n'
        'Rubric: \n'
        '    * Give a score of 1 if and only if the extracted answer and the ground truth answer '
        "are referring to the same term. It's acceptable to have equivalent grammar or form (e.g., "
        'α and alpha; $R^2_{t,h,v,m}$ and R^2_t,h,v,m). If the ground truth is a number, the '
        'extracted answer should be the number with the exact same value.\n'
        '    * Give a score of 0 if any term in the extracted answer is different from the ground '
        'truth answer, or if the extracted number is different in value from the ground truth '
        'number.\n'
        '    * When ground truth answer is "Not Applicable", the response must express "Not '
        'Applicable" to receive a score of 1.\n'
        '\n'
        '    ### Example Start ###\n'
        '    T1:\n'
        '    Response 1: The answer is 1.0\n'
        '    Ground Truth 1: 1.00\n'
        '\n'
        '    T2:\n'
        '    Response 2: By manually inspecting the plot, the final answer should be 0.\n'
        '    Ground Truth 2: Not Applicable\n'
        '\n'
        '    T3:\n'
        '    Response 3: A_v^t\n'
        '    Ground Truth 3: A^t_v\n'
        '\n'
        '    {\n'
        '        "extract_answer_T1": 1.0,\n'
        '        "score_T1": 1\n'
        '        "extract_answer_T2: 0,\n'
        '        "score_T2": 0\n'
        '        "extract_answer_T3": "A_v^t",\n'
        '        "score_T3": 1\n'
        '    }\n'
        '    ### Example End ###        \n',
 'quant': '\n'
          'Rubric:\n'
          '    * Give a score of 1 if and only if the extracted answer and the ground truth answer '
          'are numbers with the exact same value.\n'
          '    * Give a score of 0 if the extracted answer is different in value from the ground '
          'truth answer.\n'
          '    * When ground truth answer is "Not Applicable", the response must express "Not '
          'Applicable" to receive a score of 1.\n'
          '\n'
          '    ### Example Start ###\n'
          '    T1:\n'
          '    Response 1: 5\n'
          '    Ground Truth 1: 6\n'
          '\n'
          '    T2:\n'
          '    Response 2: 0\n'
          '    Ground Truth 2: Not Applicable\n'
          '\n'
          '    T3:\n'
          '    Response 3: 4\n'
          '    Ground Truth 3: 4\n'
          '\n'
          '    {\n'
          '        "extract_answer_T1": 5,\n'
          '        "score_T1": 0\n'
          '        "extract_answer_T2: 0,\n'
          '        "score_T2": 0\n'
          '        "extract_answer_T3": 4,\n'
          '        "score_T3": 1\n'
          '    }\n'
          '    ### Example End ###   \n',
 'title': '\n'
          'Rubric: \n'
          '    * Give a score of 1 if and only if the extracted answer and the ground truth answer '
          "are referring to the same term. It's acceptable to have different grammar or form "
          "(e.g., α and alpha; $R^2_{t,h,v,m}$ and R^2_t,h,v,m). It's acceptable to omit letter "
          'prefixes (e.g., (a) Increment over time and Increment over time).\n'
          '    * Give a score of 0 if any term in the extracted answer is different from the '
          'ground truth answer.\n'
          '    * When ground truth answer is "Not Applicable", the response must express "Not '
          'Applicable" to receive a score of 1.\n'
          '\n'
          '    ### Example Start ###\n'
          '    T1:\n'
          '    Response 1: The title of the plot is "The number of students in each grade".\n'
          '    Ground Truth 1: The variance of students in each grade\n'
          '\n'
          '    T2:\n'
          '    Response 2: There is no title.\n'
          '    Ground Truth 2: Not Applicable\n'
          '\n'
          '    T3:\n'
          '    Response 3: A_v^t\n'
          '    Ground Truth 3: A^t_v\n'
          '\n'
          '    {\n'
          '        "extract_answer_T1": "The number of students in each grade",\n'
          '        "score_T1": 0\n'
          '        "extract_answer_T2: "Not Applicable",\n'
          '        "score_T2": 1\n'
          '        "extract_answer_T3": "A_v^t",\n'
          '        "score_T3": 1\n'
          '    }\n'
          '    ### Example End ###        \n',
 'trend': '\n'
          'Rubric:\n'
          '    * Give a score of 1 if and only if the extracted answer and the ground truth answer '
          'share the same general trend.\n'
          '    * Give a score of 0 if the extracted answer and the ground truth answer are '
          'different in trend expression.\n'
          '\n'
          '    ### Example Start ###\n'
          '    T1:\n'
          '    Response 1: there is an increase in the data from left to right\n'
          '    Ground Truth 1: Decreases\n'
          '\n'
          '    T2:\n'
          '    Response 2: the curves move up and stay constant\n'
          '    Ground Truth 2: Increases then stabilizes\n'
          '\n'
          '    T3:\n'
          '    Response 3: Decreases\n'
          '    Ground Truth 3: Decreases then increases\n'
          '\n'
          '    {\n'
          '        "extract_answer_T1": "Increases",\n'
          '        "score_T1": 0\n'
          '        "extract_answer_T2: "Move up and stay constant",\n'
          '        "score_T2": 1\n'
          '        "extract_answer_T3": "Decreases",\n'
          '        "score_T3": 0\n'
          '    }\n'
          '    ### Example End ###\n'}

REASONING_RESP_INST = {1: '{}\n'
    '    * Your final answer must be grounded to some text that is explicitly written and relevant '
    'to the question in the chart.\n'
    '    * If you need to answer multiple terms, separate them with commas.\n'
    '    * Unless specified in the question (such as answering with a letter), you are required to '
    'answer the full names of subplots and/or labels by default.\n'
    '    ',
 2: '{}\n'
    '    * If there are options in the question, your final answer must conform to one of the '
    'options.\n'
    '    * If there are additional instructions in the question, follow them accordingly.\n'
    '    * If there are neither options nor additional instructions, you are allowed to respond '
    'with a short phrase only.\n'
    '    ',
 3: '{}\n'
    '    * Your final answer must be grounded to a number that is exlicitly written and relevant '
    "to the question in the chart, even if it's an approximate value.\n"
    '    * You are allowed to extract numbers within some text when needed.\n'
    '    ',
 4: '{}\n    {}\n    '}

REASONING_GRADING_PREFIX = ('\n'
 'You will be given a question, an ground truth answer and a model response. You need to extract '
 'the final answer from the model response, compare it with the ground truth answer, and then '
 'assign a binary score. Avoid providing explanations in your response. If there is no provided '
 'model response, please leave the extracted answer empty and give a score of 0. \n'
 '\n'
 'Your response must follow json formats with keys [extract_answer, score] where the value of the '
 'score is an interger in [0, 1]. You must follow the scoring rules:\n')

REASONING_GRADING_INST = {1: '\n'
    '    ### Rules ###\n'
    '    * Give a score of 1 if and only if the final answer and the ground truth answer are '
    "referring to the same term. It's acceptable to have different grammar or form (e.g., α and "
    "alpha; $R^2_{t,h,v,m}$ and R^2_t,h,v,m). It's also acceptable to have different orders of the "
    'terms when question asks for multiple terms.\n'
    '    * Give a score of 0 if any term (e.g., ACC+ and ACC; P-101 and P=101) is different '
    'between the final answer and the ground truth.\n'
    '\n'
    '    ### Example 1 Starts ###\n'
    '    * Question: What is the name of the curve that intersects y=\\lambda exactly three '
    'times?\n'
    '    * Ground Truth: P56962\n'
    '    * Response: There is only one curve that intersects y=\\lambda exactly three times. The '
    'name of the curve is written as P55762.\n'
    '    \n'
    '    {\n'
    '        "extracted_answer": "P55762",\n'
    '        "score": 0\n'
    '    }\n'
    '    ### Example 1 Ends ###\n'
    '\n'
    '\n'
    '    ### Example 2 Starts ###\n'
    '    * Question: What is the letter of the subplot where all bars are above 35?\n'
    '    * Ground Truth: (b)\n'
    '    * Response: The letter of the subplot where all bars are above 35 is b.\n'
    '\n'
    '    {\n'
    '        "extracted_answer": "b",\n'
    '        "score": 1\n'
    '    }\n'
    '    ### Example 2 Ends ###\n'
    '\n'
    '    ### Your Turn ###\n'
    '    * Question: <|question|>\n'
    '    * Ground Truth: <|ground_truth|>\n'
    '    * Response: <|response|>\n'
    '\n'
    '    ',
 2: '\n'
    '    ### Rules ###\n'
    '    * If there are predefined options in the question:\n'
    '        * Give a score of 1 if the final answer matches the ground truth answer exactly.\n'
    '        * Give a score of 0 if the final answer does not match the ground truth answer.\n'
    '    * If there are no predefined options in the question:\n'
    '        * Give a score of 1 if the final answer shares the same semantic meaning with the '
    'ground truth answer (e.g., "increasing then decreasing" and "moving up then down"; "converge" '
    'and "move closer together").\n'
    '        * Give a score of 0 if the final answer shares different semantic meanings from the '
    'ground truth answer (e.g., "increasing then decreasing" and "remain constant"; "converge" and '
    '"diverge").\n'
    '\n'
    '    ### Example 1 Starts ###\n'
    '    * Question: What is the trend of the red curve between t=10 and t=25?\n'
    '    * Ground Truth: increasing then decreasing\n'
    '    * Response: The red curve is increasing between t=10 and t=25.\n'
    '\n'
    '    {\n'
    '        "extracted_answer": "increasing",\n'
    '        "score": 0\n'
    '    }\n'
    '    ### Example 1 Ends ###\n'
    '\n'
    '    ### Example 2 Starts ###\n'
    '    * Question: What is the interval where the blue curve achieves the maximum value among '
    '[0, 50], [50, 100], [100, 150], and [150, 200]?\n'
    '    * Ground Truth: [50, 100]\n'
    '    * Response: The interval where the blue curve achieves the maximum value is [50, 100].\n'
    '\n'
    '    {\n'
    '        "extracted_answer": "[50, 100]",\n'
    '        "score": 1\n'
    '    }\n'
    '    ### Example 2 Ends ###\n'
    '\n'
    '    ### Your Turn ###\n'
    '    * Question: <|question|>\n'
    '    * Ground Truth: <|ground_truth|>\n'
    '    * Response: <|response|>\n'
    '\n'
    '    ',
 3: '\n'
    '    ### Rules ###\n'
    "    * Give a score of 1 if and only if the two numbers are exactly equal in values. It's "
    'acceptable to have different notations (e.g., 0.01 and 10^-2; 1500 and 1.5e3).\n'
    '    * Give a score of 0 if the two numbers are different in values.\n'
    '\n'
    '    ### Example 1 Starts ###\n'
    '    * Question: What is the value of the red curve at t=10?\n'
    '    * Ground Truth: 0.01\n'
    '    * Response: The value of the red curve at t=10 is 0.012.\n'
    '\n'
    '    {\n'
    '        "extracted_answer": "0.012",\n'
    '        "score": 0\n'
    '    }\n'
    '    ### Example 1 Ends ###\n'
    '\n'
    '    ### Example 2 Starts ###\n'
    '    * Question: What is the value of the blue curve at t=50?\n'
    '    * Ground Truth: 1500\n'
    '    * Response: The value of the blue curve at t=50 is 1.5e3.\n'
    '\n'
    '    {\n'
    '        "extracted_answer": "1.5e3",\n'
    '        "score": 1\n'
    '    }\n'
    '    ### Example 2 Ends ###\n'
    '\n'
    '    ### Your Turn ###\n'
    '    * Question: <|question|>\n'
    '    * Ground Truth: <|ground_truth|>\n'
    '    * Response: <|response|>\n'
    '\n'
    '    ',
 4: '\n'
    '    ### Rules ###\n'
    "    * Give a score of 1 if and only if the two numbers are exactly equal in values. It's "
    'acceptable to have different notations (e.g., 0.01 and 10^-2; 1500 and 1.5e3).\n'
    '    * Give a score of 0 if the two numbers are different in values.\n'
    '\n'
    '    ### Example 1 Starts ###\n'
    '    * Question: What is the value of the red curve at t=10?\n'
    '    * Ground Truth: 0.01\n'
    '    * Response: The value of the red curve at t=10 is 0.012.\n'
    '\n'
    '    {\n'
    '        "extracted_answer": "0.012",\n'
    '        "score": 0\n'
    '    }\n'
    '    ### Example 1 Ends ###\n'
    '\n'
    '    ### Example 2 Starts ###\n'
    '    * Question: What is the value of the blue curve at t=50?\n'
    '    * Ground Truth: 1500\n'
    '    * Response: The value of the blue curve at t=50 is 1.5e3.\n'
    '\n'
    '    {\n'
    '        "extracted_answer": "1.5e3",\n'
    '        "score": 1\n'
    '    }\n'
    '    ### Example 2 Ends ###\n'
    '\n'
    '    ### Your Turn ###\n'
    '    * Question: <|question|>\n'
    '    * Ground Truth: <|ground_truth|>\n'
    '    * Response: <|response|>\n'
    '\n'
    '    '}

# qid -> rubric group, transcribed from CharXiv's `descriptive_utils.get_rubric`.
DESCRIPTIVE_QID_RUBRIC = {
    **{q: "title"  for q in [1]},
    **{q: "ocr"    for q in [2, 3, 4, 5, 6, 7]},
    **{q: "quant"  for q in [8, 9, 10, 12, 14, 15, 17, 19]},
    **{q: "bool"   for q in [11]},
    **{q: "enum"   for q in [13]},
    **{q: "trend"  for q in [16]},
    **{q: "layout" for q in [18]},
}


def rubric_for(qid: int) -> str:
    """The official per-question-type judging rubric."""
    return DESCRIPTIVE_GRADING_ICL[DESCRIPTIVE_QID_RUBRIC[int(qid)]]


def get_number_instruction(answer: str) -> str:
    """Transcribed verbatim from CharXiv's `reasoning_utils.get_number_instruction`.

    Reasoning type 4 ("number-in-general") is the only one whose prompt depends
    on the gold answer: CharXiv tells the model how many decimal places to give,
    derived from the answer itself. Omitting it makes a correct value fail the
    grader on formatting alone.
    """
    base = str(answer).split(".")
    whole, decimal = base[0], None if len(base) == 1 else base[1]
    if whole is not None and decimal is None:
        return "* Your final answer must be an exact integer."
    if whole is not None and decimal is not None:
        return f"* Your final answer must be a number with {len(decimal)} decimal places."
    raise ValueError(f"Invalid answer: {answer}")


def reasoning_question(query: str, inst_category: int, answer: str) -> str:
    """CharXiv's official reasoning prompt: the query wrapped by answer type.

    Categories: 1 text-in-chart, 2 text-in-general, 3 number-in-chart,
    4 number-in-general (needs the decimal-place instruction).
    """
    c = int(inst_category)
    if c in (1, 2, 3):
        return REASONING_RESP_INST[c].format(query)
    if c == 4:
        return REASONING_RESP_INST[4].format(query, get_number_instruction(answer))
    raise ValueError(f"Invalid instruction category: {inst_category}")
