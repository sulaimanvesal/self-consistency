"""Built-in sample dataset: 12 handwritten GSM8K-style arithmetic word problems.

No downloads, no network — everything ships with the package so the demo and
evaluation run fully offline.
"""

SAMPLE_PROBLEMS = [
    {
        "id": "q01",
        "question": "Janet has 3 apples. She buys 4 more apples, then gives 2 apples to her friend. How many apples does Janet have now?",
        "gold": "5",
        "steps": [
            ("3 + 4 = 7 apples after buying", "7"),
            ("7 - 2 = 5 apples after giving 2 away", "5"),
        ],
        "distractor": "7",
    },
    {
        "id": "q02",
        "question": "A shirt costs $20 and is on sale for 25% off. What is the sale price of the shirt in dollars?",
        "gold": "15",
        "steps": [
            ("25% of $20 = $5 discount", "5"),
            ("$20 - $5 = $15 sale price", "15"),
        ],
        "distractor": "5",
    },
    {
        "id": "q03",
        "question": "Tom runs 3 miles every day for 5 days. How many miles does he run in total?",
        "gold": "15",
        "steps": [
            ("3 miles/day x 5 days = 15 miles total", "15"),
        ],
        "distractor": "8",
    },
    {
        "id": "q04",
        "question": "There are 24 chocolates in a box. Six are eaten, and the rest are shared equally among 3 friends. How many chocolates does each friend get?",
        "gold": "6",
        "steps": [
            ("24 - 6 = 18 chocolates left", "18"),
            ("18 / 3 = 6 chocolates per friend", "6"),
        ],
        "distractor": "8",
    },
    {
        "id": "q05",
        "question": "A bakery bakes 48 muffins. It sells 30 of them, then packs the remaining muffins into boxes of 6. How many boxes are needed?",
        "gold": "3",
        "steps": [
            ("48 - 30 = 18 muffins remaining", "18"),
            ("18 / 6 = 3 boxes needed", "3"),
        ],
        "distractor": "8",
    },
    {
        "id": "q06",
        "question": "Lisa has $50. She buys a book for $12 and a pen for $3. How much money does she have left?",
        "gold": "35",
        "steps": [
            ("$50 - $12 = $38 after the book", "38"),
            ("$38 - $3 = $35 left", "35"),
        ],
        "distractor": "41",
    },
    {
        "id": "q07",
        "question": "A factory has 8 workers. Each worker makes 12 widgets per day, and they work for 3 days. How many widgets are made in total?",
        "gold": "288",
        "steps": [
            ("12 widgets/day x 3 days = 36 widgets per worker", "36"),
            ("36 x 8 workers = 288 widgets total", "288"),
        ],
        "distractor": "96",
    },
    {
        "id": "q08",
        "question": "A train travels at 60 miles per hour for 2.5 hours. How many miles does it travel?",
        "gold": "150",
        "steps": [
            ("60 mph x 2.5 hours = 150 miles", "150"),
        ],
        "distractor": "24",
    },
    {
        "id": "q09",
        "question": "There are 30 students in a class. One third of them are girls. How many boys are in the class?",
        "gold": "20",
        "steps": [
            ("30 / 3 = 10 girls", "10"),
            ("30 - 10 = 20 boys", "20"),
        ],
        "distractor": "10",
    },
    {
        "id": "q10",
        "question": "Sarah saves $15 each week for 8 weeks to buy a bike that costs $100. How much extra money will she have beyond the price of the bike?",
        "gold": "20",
        "steps": [
            ("$15 x 8 weeks = $120 saved", "120"),
            ("$120 - $100 = $20 extra", "20"),
        ],
        "distractor": "120",
    },
    {
        "id": "q11",
        "question": "A rectangular garden is 12 feet long and 5 feet wide. How many feet of fence are needed to enclose it completely?",
        "gold": "34",
        "steps": [
            ("perimeter = 2 x (12 + 5)", "34"),
            ("2 x 17 = 34 feet of fence", "34"),
        ],
        "distractor": "60",
    },
    {
        "id": "q12",
        "question": "Movie tickets cost $14 for adults and $9 for children. A family buys 2 adult tickets and 3 child tickets. What is the total cost in dollars?",
        "gold": "55",
        "steps": [
            ("2 x $14 = $28 for adults", "28"),
            ("3 x $9 = $27 for children", "27"),
            ("$28 + $27 = $55 total", "55"),
        ],
        "distractor": "69",
    },
]
