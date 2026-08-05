import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import joblib


sector_config = {
    "Textiles and Apparel": {
        "category_weight": 0.06,

        "peak_months": [9, 10, 11],
        "lean_months": [4, 5],

        "peak_multiplier": (1.30, 1.60),
        "lean_multiplier": (0.65, 0.85),

        "subcategories": {

            "Handlooms": {
                "weight": 0.25,

                "employee_range": {
                    "micro": (3, 10),
                    "small": (10, 30),
                    "medium": (30, 80)
                },

                "salary_range": (10000, 18000),

                "average_ticket_size": (500, 5000),

                "b2b_share": (0.20, 0.40),

                "upi_share": (0.25, 0.45),
                "cash_share": (0.35, 0.55),
                "bank_transfer_share": (0.10, 0.25),

                "gst_rate": [5, 12],

                "working_capital_days": (30, 60),

                "export_probability": 0.20,
                "e_invoice_probability": 0.03,

                "seasonality_strength": (0.25, 0.45),

                "monthly_growth_rate": (-0.02, 0.04)
            },

            "Power looms": {
                "weight": 0.25,

                "employee_range": {
                    "micro": (5, 20),
                    "small": (20, 60),
                    "medium": (60, 150)
                },

                "salary_range": (12000, 22000),

                "average_ticket_size": (2000, 20000),

                "b2b_share": (0.70, 0.90),

                "upi_share": (0.15, 0.30),
                "cash_share": (0.15, 0.30),
                "bank_transfer_share": (0.45, 0.65),

                "gst_rate": [5, 12, 18],

                "working_capital_days": (30, 45),

                "export_probability": 0.10,
                "e_invoice_probability": 0.10,

                "seasonality_strength": (0.15, 0.30),

                "monthly_growth_rate": (-0.02, 0.05)
            },

            "Knitting": {
                "weight": 0.15,

                "employee_range": {
                    "micro": (5, 15),
                    "small": (15, 50),
                    "medium": (50, 120)
                },

                "salary_range": (10000, 18000),

                "average_ticket_size": (1000, 10000),

                "b2b_share": (0.50, 0.70),

                "upi_share": (0.20, 0.35),
                "cash_share": (0.20, 0.40),
                "bank_transfer_share": (0.30, 0.50),

                "gst_rate": [5, 12],

                "working_capital_days": (30, 50),

                "export_probability": 0.12,
                "e_invoice_probability": 0.08,

                "seasonality_strength": (0.15, 0.30),

                "monthly_growth_rate": (-0.02, 0.05)
            },

            "Garment manufacturing units": {
                "weight": 0.35,

                "employee_range": {
                    "micro": (10, 30),
                    "small": (30, 100),
                    "medium": (100, 300)
                },

                "salary_range": (14000, 25000),

                "average_ticket_size": (3000, 30000),

                "b2b_share": (0.55, 0.80),

                "upi_share": (0.15, 0.30),
                "cash_share": (0.10, 0.25),
                "bank_transfer_share": (0.45, 0.65),

                "gst_rate": [5, 12],

                "working_capital_days": (35, 60),

                "export_probability": 0.35,
                "e_invoice_probability": 0.25,

                "seasonality_strength": (0.20, 0.40),

                "monthly_growth_rate": (-0.01, 0.05)
            }
        }
    },

    "Food Processing": {
        "category_weight": 0.07,

        "peak_months": [10, 11, 12],
        "lean_months": [7],

        "peak_multiplier": (1.20, 1.50),
        "lean_multiplier": (0.70, 0.90),

        "subcategories": {

            "Agricultural product processing": {
                "weight": 0.40,

                "employee_range": {
                    "micro": (5, 20),
                    "small": (20, 80),
                    "medium": (80, 250)
                },

                "salary_range": (12000, 22000),

                "average_ticket_size": (3000, 40000),

                "b2b_share": (0.70, 0.90),

                "upi_share": (0.10, 0.25),
                "cash_share": (0.10, 0.25),
                "bank_transfer_share": (0.60, 0.80),

                "gst_rate": [5, 12],

                "working_capital_days": (45, 90),

                "export_probability": 0.15,
                "e_invoice_probability": 0.15,

                "seasonality_strength": (0.30, 0.50),

                "monthly_growth_rate": (-0.02, 0.05)
            },

            "Dairy products": {
                "weight": 0.25,

                "employee_range": {
                    "micro": (8, 25),
                    "small": (25, 80),
                    "medium": (80, 250)
                },

                "salary_range": (13000, 23000),

                "average_ticket_size": (500, 15000),

                "b2b_share": (0.50, 0.75),

                "upi_share": (0.20, 0.40),
                "cash_share": (0.15, 0.30),
                "bank_transfer_share": (0.35, 0.60),

                "gst_rate": [5, 12],

                "working_capital_days": (15, 35),

                "export_probability": 0.08,
                "e_invoice_probability": 0.12,

                "seasonality_strength": (0.10, 0.25),

                "monthly_growth_rate": (-0.01, 0.04)
            },

            "Confectionery": {
                "weight": 0.10,

                "employee_range": {
                    "micro": (3, 15),
                    "small": (15, 50),
                    "medium": (50, 150)
                },

                "salary_range": (12000, 20000),

                "average_ticket_size": (300, 5000),

                "b2b_share": (0.40, 0.70),

                "upi_share": (0.30, 0.50),
                "cash_share": (0.20, 0.40),
                "bank_transfer_share": (0.20, 0.40),

                "gst_rate": [5, 12, 18],

                "working_capital_days": (20, 45),

                "export_probability": 0.05,
                "e_invoice_probability": 0.08,

                "seasonality_strength": (0.35, 0.60),

                "monthly_growth_rate": (-0.02, 0.06)
            },

            "Packaged foods": {
                "weight": 0.25,

                "employee_range": {
                    "micro": (10, 30),
                    "small": (30, 100),
                    "medium": (100, 300)
                },

                "salary_range": (14000, 25000),

                "average_ticket_size": (2000, 30000),

                "b2b_share": (0.65, 0.90),

                "upi_share": (0.15, 0.30),
                "cash_share": (0.10, 0.25),
                "bank_transfer_share": (0.50, 0.70),

                "gst_rate": [5, 12, 18],

                "working_capital_days": (30, 60),

                "export_probability": 0.20,
                "e_invoice_probability": 0.20,

                "seasonality_strength": (0.20, 0.40),

                "monthly_growth_rate": (-0.01, 0.05)
            }
        }
    },

    "Leather and Leather Products": {
        "category_weight": 0.02,

        "peak_months": [9, 10, 11],
        "lean_months": [5, 6],

        "peak_multiplier": (1.25, 1.50),
        "lean_multiplier": (0.70, 0.90),

        "subcategories": {

            "Footwear": {
                "weight": 0.50,

                "employee_range": {
                    "micro": (5, 20),
                    "small": (20, 80),
                    "medium": (80, 250)
                },

                "salary_range": (12000, 22000),

                "average_ticket_size": (1000, 12000),

                "b2b_share": (0.55, 0.80),

                "upi_share": (0.20, 0.40),
                "cash_share": (0.15, 0.30),
                "bank_transfer_share": (0.35, 0.60),

                "gst_rate": [5, 12, 18],

                "working_capital_days": (30, 60),

                "export_probability": 0.20,
                "e_invoice_probability": 0.12,

                "seasonality_strength": (0.20, 0.40),

                "monthly_growth_rate": (-0.02, 0.05)
            },

            "Bags": {
                "weight": 0.20,

                "employee_range": {
                    "micro": (4, 15),
                    "small": (15, 50),
                    "medium": (50, 150)
                },

                "salary_range": (12000, 22000),

                "average_ticket_size": (1500, 15000),

                "b2b_share": (0.60, 0.85),

                "upi_share": (0.20, 0.35),
                "cash_share": (0.15, 0.25),
                "bank_transfer_share": (0.45, 0.60),

                "gst_rate": [12, 18],

                "working_capital_days": (35, 70),

                "export_probability": 0.30,
                "e_invoice_probability": 0.15,

                "seasonality_strength": (0.20, 0.35),

                "monthly_growth_rate": (-0.02, 0.05)
            },

            "Belts": {
                "weight": 0.15,

                "employee_range": {
                    "micro": (3, 12),
                    "small": (12, 40),
                    "medium": (40, 120)
                },

                "salary_range": (12000, 20000),

                "average_ticket_size": (500, 5000),

                "b2b_share": (0.70, 0.90),

                "upi_share": (0.10, 0.25),
                "cash_share": (0.15, 0.30),
                "bank_transfer_share": (0.50, 0.70),

                "gst_rate": [12, 18],

                "working_capital_days": (30, 60),

                "export_probability": 0.25,
                "e_invoice_probability": 0.18,

                "seasonality_strength": (0.15, 0.30),

                "monthly_growth_rate": (-0.02, 0.04)
            },

            "Accessories manufacturing": {
                "weight": 0.15,

                "employee_range": {
                    "micro": (3, 15),
                    "small": (15, 50),
                    "medium": (50, 180)
                },

                "salary_range": (12000, 22000),

                "average_ticket_size": (800, 8000),

                "b2b_share": (0.60, 0.85),

                "upi_share": (0.15, 0.30),
                "cash_share": (0.15, 0.25),
                "bank_transfer_share": (0.45, 0.65),

                "gst_rate": [12, 18],

                "working_capital_days": (35, 65),

                "export_probability": 0.35,
                "e_invoice_probability": 0.18,

                "seasonality_strength": (0.20, 0.35),

                "monthly_growth_rate": (-0.02, 0.05)
            }
        }
    },

    "Chemical Products": {
        "category_weight": 0.02,

        "peak_months": [1, 2, 3],
        "lean_months": [7, 8],

        "peak_multiplier": (1.15, 1.35),
        "lean_multiplier": (0.80, 0.95),

        "subcategories": {

            "Chemicals": {
                "weight": 0.45,

                "employee_range": {
                    "micro": (5, 20),
                    "small": (20, 80),
                    "medium": (80, 250)
                },

                "salary_range": (18000, 35000),

                "average_ticket_size": (5000, 100000),

                "b2b_share": (0.85, 0.98),

                "upi_share": (0.02, 0.08),
                "cash_share": (0.02, 0.08),
                "bank_transfer_share": (0.85, 0.95),

                "gst_rate": [18],

                "working_capital_days": (45, 90),

                "export_probability": 0.20,
                "e_invoice_probability": 0.45,

                "seasonality_strength": (0.05, 0.15),

                "monthly_growth_rate": (-0.01, 0.04)
            },

            "Dyes": {
                "weight": 0.25,

                "employee_range": {
                    "micro": (5, 15),
                    "small": (15, 60),
                    "medium": (60, 180)
                },

                "salary_range": (18000, 32000),

                "average_ticket_size": (3000, 60000),

                "b2b_share": (0.90, 0.99),

                "upi_share": (0.01, 0.05),
                "cash_share": (0.01, 0.05),
                "bank_transfer_share": (0.90, 0.98),

                "gst_rate": [18],

                "working_capital_days": (45, 90),

                "export_probability": 0.30,
                "e_invoice_probability": 0.40,

                "seasonality_strength": (0.10, 0.20),

                "monthly_growth_rate": (-0.01, 0.04)
            },

            "Paints": {
                "weight": 0.30,

                "employee_range": {
                    "micro": (5, 20),
                    "small": (20, 70),
                    "medium": (70, 200)
                },

                "salary_range": (17000, 32000),

                "average_ticket_size": (2000, 50000),

                "b2b_share": (0.70, 0.90),

                "upi_share": (0.05, 0.15),
                "cash_share": (0.05, 0.15),
                "bank_transfer_share": (0.75, 0.90),

                "gst_rate": [18],

                "working_capital_days": (40, 80),

                "export_probability": 0.15,
                "e_invoice_probability": 0.35,

                "seasonality_strength": (0.20, 0.35),

                "monthly_growth_rate": (-0.01, 0.05)
            }
        }
    },
    "Pharmaceuticals": {
        "category_weight": 0.01,

        "peak_months": [1, 7],
        "lean_months": [4],

        "peak_multiplier": (1.10, 1.30),
        "lean_multiplier": (0.85, 0.95),

        "subcategories": {

            "Tablets": {
                "weight": 0.45,

                "employee_range": {
                    "micro": (10, 30),
                    "small": (30, 120),
                    "medium": (120, 350)
                },

                "salary_range": (20000, 40000),

                "average_ticket_size": (5000, 100000),

                "b2b_share": (0.90, 0.99),

                "upi_share": (0.01, 0.05),
                "cash_share": (0.01, 0.05),
                "bank_transfer_share": (0.90, 0.98),

                "gst_rate": [5, 12],

                "working_capital_days": (45, 90),

                "export_probability": 0.30,
                "e_invoice_probability": 0.60,

                "seasonality_strength": (0.05, 0.15),

                "monthly_growth_rate": (-0.01, 0.04)
            },

            "Capsules": {
                "weight": 0.30,

                "employee_range": {
                    "micro": (8, 25),
                    "small": (25, 100),
                    "medium": (100, 300)
                },

                "salary_range": (20000, 38000),

                "average_ticket_size": (5000, 80000),

                "b2b_share": (0.90, 0.99),

                "upi_share": (0.01, 0.05),
                "cash_share": (0.01, 0.05),
                "bank_transfer_share": (0.90, 0.98),

                "gst_rate": [5, 12],

                "working_capital_days": (45, 90),

                "export_probability": 0.25,
                "e_invoice_probability": 0.55,

                "seasonality_strength": (0.05, 0.15),

                "monthly_growth_rate": (-0.01, 0.04)
            },

            "Syrups": {
                "weight": 0.25,

                "employee_range": {
                    "micro": (8, 20),
                    "small": (20, 80),
                    "medium": (80, 250)
                },

                "salary_range": (18000, 35000),

                "average_ticket_size": (2000, 50000),

                "b2b_share": (0.80, 0.95),

                "upi_share": (0.02, 0.08),
                "cash_share": (0.02, 0.08),
                "bank_transfer_share": (0.85, 0.95),

                "gst_rate": [12],

                "working_capital_days": (30, 75),

                "export_probability": 0.15,
                "e_invoice_probability": 0.45,

                "seasonality_strength": (0.15, 0.30),

                "monthly_growth_rate": (-0.01, 0.05)
            }
        }
    },
    "Engineering Goods": {
        "category_weight": 0.04,

        "peak_months": [1, 2, 3],
        "lean_months": [6, 7],

        "peak_multiplier": (1.15, 1.40),
        "lean_multiplier": (0.80, 0.95),

        "subcategories": {

            "Machinery": {
                "weight": 0.40,

                "employee_range": {
                    "micro": (8, 25),
                    "small": (25, 100),
                    "medium": (100, 350)
                },

                "salary_range": (20000, 40000),

                "average_ticket_size": (50000, 1000000),

                "b2b_share": (0.95, 0.99),

                "upi_share": (0.00, 0.03),
                "cash_share": (0.00, 0.03),
                "bank_transfer_share": (0.94, 1.00),

                "gst_rate": [18],

                "working_capital_days": (60, 120),

                "export_probability": 0.30,
                "e_invoice_probability": 0.70,

                "seasonality_strength": (0.05, 0.15),

                "monthly_growth_rate": (-0.01, 0.05)
            },

            "Equipment": {
                "weight": 0.25,

                "employee_range": {
                    "micro": (6, 20),
                    "small": (20, 80),
                    "medium": (80, 250)
                },

                "salary_range": (18000, 35000),

                "average_ticket_size": (10000, 500000),

                "b2b_share": (0.90, 0.98),

                "upi_share": (0.01, 0.05),
                "cash_share": (0.01, 0.04),
                "bank_transfer_share": (0.90, 0.98),

                "gst_rate": [18],

                "working_capital_days": (45, 90),

                "export_probability": 0.20,
                "e_invoice_probability": 0.60,

                "seasonality_strength": (0.08, 0.18),

                "monthly_growth_rate": (-0.01, 0.05)
            },

            "Metal product manufacturing": {
                "weight": 0.35,

                "employee_range": {
                    "micro": (10, 30),
                    "small": (30, 120),
                    "medium": (120, 400)
                },

                "salary_range": (18000, 32000),

                "average_ticket_size": (5000, 200000),

                "b2b_share": (0.85, 0.98),

                "upi_share": (0.02, 0.08),
                "cash_share": (0.02, 0.08),
                "bank_transfer_share": (0.85, 0.95),

                "gst_rate": [18],

                "working_capital_days": (45, 90),

                "export_probability": 0.18,
                "e_invoice_probability": 0.50,

                "seasonality_strength": (0.10, 0.20),

                "monthly_growth_rate": (-0.02, 0.05)
            }
        }
    },
    "Rubber and Plastic Products": {
        "category_weight": 0.02,

        "peak_months": [9, 10],
        "lean_months": [6],

        "peak_multiplier": (1.15, 1.40),
        "lean_multiplier": (0.80, 0.95),

        "subcategories": {

            "Rubber items": {
                "weight": 0.35,

                "employee_range": {
                    "micro": (5, 20),
                    "small": (20, 80),
                    "medium": (80, 250)
                },

                "salary_range": (15000, 30000),

                "average_ticket_size": (2000, 50000),

                "b2b_share": (0.75, 0.95),

                "upi_share": (0.05, 0.15),
                "cash_share": (0.05, 0.15),
                "bank_transfer_share": (0.75, 0.90),

                "gst_rate": [12, 18],

                "working_capital_days": (40, 75),

                "export_probability": 0.20,
                "e_invoice_probability": 0.35,

                "seasonality_strength": (0.10, 0.25),

                "monthly_growth_rate": (-0.01, 0.05)
            },

            "Plastic goods": {
                "weight": 0.65,

                "employee_range": {
                    "micro": (5, 25),
                    "small": (25, 100),
                    "medium": (100, 300)
                },

                "salary_range": (15000, 28000),

                "average_ticket_size": (1000, 30000),

                "b2b_share": (0.65, 0.90),

                "upi_share": (0.10, 0.25),
                "cash_share": (0.05, 0.20),
                "bank_transfer_share": (0.60, 0.80),

                "gst_rate": [12, 18],

                "working_capital_days": (30, 60),

                "export_probability": 0.12,
                "e_invoice_probability": 0.30,

                "seasonality_strength": (0.15, 0.30),

                "monthly_growth_rate": (-0.02, 0.05)
            }
        }
    },
    "Electrical and Electronics": {
        "category_weight": 0.03,

        "peak_months": [10, 11],
        "lean_months": [2],

        "peak_multiplier": (1.20, 1.50),
        "lean_multiplier": (0.80, 0.95),

        "subcategories": {

            "Electrical appliances": {
                "weight": 0.30,

                "employee_range": {
                    "micro": (5, 20),
                    "small": (20, 80),
                    "medium": (80, 250)
                },

                "salary_range": (18000, 35000),

                "average_ticket_size": (3000, 50000),

                "b2b_share": (0.50, 0.75),

                "upi_share": (0.15, 0.30),
                "cash_share": (0.05, 0.20),
                "bank_transfer_share": (0.55, 0.75),

                "gst_rate": [18],

                "working_capital_days": (35, 70),

                "export_probability": 0.12,
                "e_invoice_probability": 0.35,

                "seasonality_strength": (0.25, 0.45),

                "monthly_growth_rate": (-0.02, 0.05)
            },

            "Electrical components": {
                "weight": 0.40,

                "employee_range": {
                    "micro": (8, 25),
                    "small": (25, 100),
                    "medium": (100, 300)
                },

                "salary_range": (20000, 38000),

                "average_ticket_size": (5000, 100000),

                "b2b_share": (0.85, 0.98),

                "upi_share": (0.02, 0.08),
                "cash_share": (0.02, 0.08),
                "bank_transfer_share": (0.85, 0.95),

                "gst_rate": [18],

                "working_capital_days": (45, 90),

                "export_probability": 0.25,
                "e_invoice_probability": 0.55,

                "seasonality_strength": (0.10, 0.20),

                "monthly_growth_rate": (-0.01, 0.05)
            },

            "Electrical gadgets manufacturing": {
                "weight": 0.30,

                "employee_range": {
                    "micro": (10, 30),
                    "small": (30, 120),
                    "medium": (120, 350)
                },

                "salary_range": (20000, 40000),

                "average_ticket_size": (2000, 40000),

                "b2b_share": (0.70, 0.90),

                "upi_share": (0.08, 0.20),
                "cash_share": (0.03, 0.10),
                "bank_transfer_share": (0.70, 0.88),

                "gst_rate": [18],

                "working_capital_days": (40, 80),

                "export_probability": 0.30,
                "e_invoice_probability": 0.50,

                "seasonality_strength": (0.20, 0.35),

                "monthly_growth_rate": (-0.01, 0.06)
            }
        }
    },
    "Handicrafts and Artisan Products": {
        "category_weight": 0.03,

        "peak_months": [10, 11, 12],
        "lean_months": [6, 7],

        "peak_multiplier": (1.30, 1.60),
        "lean_multiplier": (0.65, 0.85),

        "subcategories": {

            "Traditional handmade products": {
                "weight": 0.75,

                "employee_range": {
                    "micro": (2, 10),
                    "small": (10, 30),
                    "medium": (30, 80)
                },

                "salary_range": (10000, 18000),

                "average_ticket_size": (300, 5000),

                "b2b_share": (0.20, 0.45),

                "upi_share": (0.30, 0.50),
                "cash_share": (0.30, 0.50),
                "bank_transfer_share": (0.10, 0.30),

                "gst_rate": [5, 12],

                "working_capital_days": (20, 45),

                "export_probability": 0.30,
                "e_invoice_probability": 0.05,

                "seasonality_strength": (0.35, 0.60),

                "monthly_growth_rate": (-0.03, 0.05)
            },

            "Contemporary handmade products": {
                "weight": 0.25,

                "employee_range": {
                    "micro": (3, 15),
                    "small": (15, 40),
                    "medium": (40, 100)
                },

                "salary_range": (12000, 22000),

                "average_ticket_size": (500, 10000),

                "b2b_share": (0.35, 0.60),

                "upi_share": (0.40, 0.60),
                "cash_share": (0.15, 0.35),
                "bank_transfer_share": (0.20, 0.40),

                "gst_rate": [5, 12],

                "working_capital_days": (20, 40),

                "export_probability": 0.20,
                "e_invoice_probability": 0.10,

                "seasonality_strength": (0.25, 0.45),

                "monthly_growth_rate": (-0.02, 0.06)
            }
        }
    },
    "Paper Products": {
        "category_weight": 0.02,

        "peak_months": [5, 6],
        "lean_months": [1],

        "peak_multiplier": (1.15, 1.35),
        "lean_multiplier": (0.85, 0.95),

        "subcategories": {

            "Stationary items": {
                "weight": 0.35,

                "employee_range": {
                    "micro": (3, 12),
                    "small": (12, 40),
                    "medium": (40, 120)
                },

                "salary_range": (12000, 22000),

                "average_ticket_size": (200, 5000),

                "b2b_share": (0.35, 0.60),

                "upi_share": (0.30, 0.50),
                "cash_share": (0.20, 0.40),
                "bank_transfer_share": (0.20, 0.40),

                "gst_rate": [12, 18],

                "working_capital_days": (20, 45),

                "export_probability": 0.05,
                "e_invoice_probability": 0.10,

                "seasonality_strength": (0.20, 0.40),

                "monthly_growth_rate": (-0.02, 0.05)
            },

            "Packaging materials production": {
                "weight": 0.65,

                "employee_range": {
                    "micro": (5, 20),
                    "small": (20, 80),
                    "medium": (80, 250)
                },

                "salary_range": (15000, 28000),

                "average_ticket_size": (3000, 75000),

                "b2b_share": (0.85, 0.98),

                "upi_share": (0.02, 0.08),
                "cash_share": (0.02, 0.08),
                "bank_transfer_share": (0.85, 0.95),

                "gst_rate": [12, 18],

                "working_capital_days": (30, 60),

                "export_probability": 0.15,
                "e_invoice_probability": 0.35,

                "seasonality_strength": (0.10, 0.25),

                "monthly_growth_rate": (-0.01, 0.05)
            }
        }
    },
    "Agro-based Industries": {
        "category_weight": 0.04,

        "peak_months": [10, 11],
        "lean_months": [5],

        "peak_multiplier": (1.20, 1.45),
        "lean_multiplier": (0.75, 0.90),

        "subcategories": {

            "Dairy": {
                "weight": 0.45,

                "employee_range": {
                    "micro": (5, 20),
                    "small": (20, 80),
                    "medium": (80, 250)
                },

                "salary_range": (14000, 25000),

                "average_ticket_size": (500, 10000),

                "b2b_share": (0.45, 0.70),

                "upi_share": (0.20, 0.40),
                "cash_share": (0.15, 0.35),
                "bank_transfer_share": (0.35, 0.55),

                "gst_rate": [5, 12],

                "working_capital_days": (15, 35),

                "export_probability": 0.08,
                "e_invoice_probability": 0.12,

                "seasonality_strength": (0.10, 0.20),

                "monthly_growth_rate": (-0.01, 0.04)
            },

            "Honey processing": {
                "weight": 0.10,

                "employee_range": {
                    "micro": (3, 12),
                    "small": (12, 40),
                    "medium": (40, 100)
                },

                "salary_range": (13000, 22000),

                "average_ticket_size": (500, 8000),

                "b2b_share": (0.50, 0.75),

                "upi_share": (0.25, 0.45),
                "cash_share": (0.15, 0.30),
                "bank_transfer_share": (0.25, 0.45),

                "gst_rate": [5, 12],

                "working_capital_days": (25, 50),

                "export_probability": 0.30,
                "e_invoice_probability": 0.10,

                "seasonality_strength": (0.20, 0.35),

                "monthly_growth_rate": (-0.02, 0.05)
            },

            "Pickles": {
                "weight": 0.25,

                "employee_range": {
                    "micro": (3, 15),
                    "small": (15, 50),
                    "medium": (50, 150)
                },

                "salary_range": (12000, 22000),

                "average_ticket_size": (300, 5000),

                "b2b_share": (0.40, 0.70),

                "upi_share": (0.30, 0.50),
                "cash_share": (0.20, 0.35),
                "bank_transfer_share": (0.20, 0.40),

                "gst_rate": [5, 12],

                "working_capital_days": (20, 45),

                "export_probability": 0.15,
                "e_invoice_probability": 0.08,

                "seasonality_strength": (0.30, 0.50),

                "monthly_growth_rate": (-0.02, 0.05)
            },

            "Jam production": {
                "weight": 0.20,

                "employee_range": {
                    "micro": (3, 15),
                    "small": (15, 50),
                    "medium": (50, 150)
                },

                "salary_range": (13000, 23000),

                "average_ticket_size": (400, 6000),

                "b2b_share": (0.50, 0.75),

                "upi_share": (0.25, 0.45),
                "cash_share": (0.15, 0.30),
                "bank_transfer_share": (0.25, 0.45),

                "gst_rate": [5, 12],

                "working_capital_days": (25, 50),

                "export_probability": 0.20,
                "e_invoice_probability": 0.10,

                "seasonality_strength": (0.25, 0.45),

                "monthly_growth_rate": (-0.02, 0.05)
            }
        }
    },
    "Furniture and Wood Products": {
        "category_weight": 0.03,

        "peak_months": [10, 11, 12],
        "lean_months": [6],

        "peak_multiplier": (1.25, 1.50),
        "lean_multiplier": (0.75, 0.90),

        "subcategories": {

            "Wooden": {
                "weight": 0.60,

                "employee_range": {
                    "micro": (5, 20),
                    "small": (20, 80),
                    "medium": (80, 250)
                },

                "salary_range": (15000, 30000),

                "average_ticket_size": (5000, 100000),

                "b2b_share": (0.55, 0.80),

                "upi_share": (0.10, 0.25),
                "cash_share": (0.10, 0.25),
                "bank_transfer_share": (0.55, 0.75),

                "gst_rate": [12, 18],

                "working_capital_days": (45, 90),

                "export_probability": 0.20,
                "e_invoice_probability": 0.25,

                "seasonality_strength": (0.25, 0.45),

                "monthly_growth_rate": (-0.02, 0.05)
            },

            "Bamboo": {
                "weight": 0.20,

                "employee_range": {
                    "micro": (3, 12),
                    "small": (12, 40),
                    "medium": (40, 120)
                },

                "salary_range": (12000, 22000),

                "average_ticket_size": (1000, 30000),

                "b2b_share": (0.35, 0.60),

                "upi_share": (0.25, 0.45),
                "cash_share": (0.20, 0.40),
                "bank_transfer_share": (0.20, 0.40),

                "gst_rate": [12],

                "working_capital_days": (25, 60),

                "export_probability": 0.25,
                "e_invoice_probability": 0.08,

                "seasonality_strength": (0.30, 0.50),

                "monthly_growth_rate": (-0.02, 0.06)
            },

            "Cane": {
                "weight": 0.20,

                "employee_range": {
                    "micro": (3, 10),
                    "small": (10, 35),
                    "medium": (35, 100)
                },

                "salary_range": (12000, 22000),

                "average_ticket_size": (1000, 25000),

                "b2b_share": (0.30, 0.55),

                "upi_share": (0.30, 0.50),
                "cash_share": (0.20, 0.40),
                "bank_transfer_share": (0.15, 0.35),

                "gst_rate": [12],

                "working_capital_days": (20, 50),

                "export_probability": 0.30,
                "e_invoice_probability": 0.05,

                "seasonality_strength": (0.35, 0.55),

                "monthly_growth_rate": (-0.02, 0.06)
            }
        }
    },
    "Gems and Jewellery": {
        "category_weight": 0.03,

        "peak_months": [10, 11, 12],
        "lean_months": [5, 6],

        "peak_multiplier": (1.30, 1.60),
        "lean_multiplier": (0.70, 0.90),

        "subcategories": {

            "Crafting of jewellery": {
                "weight": 0.60,

                "employee_range": {
                    "micro": (3, 12),
                    "small": (12, 40),
                    "medium": (40, 120)
                },

                "salary_range": (20000, 45000),

                "average_ticket_size": (5000, 200000),

                "b2b_share": (0.55, 0.80),

                "upi_share": (0.10, 0.25),
                "cash_share": (0.10, 0.25),
                "bank_transfer_share": (0.55, 0.75),

                "gst_rate": [3],

                "working_capital_days": (60, 120),

                "export_probability": 0.25,
                "e_invoice_probability": 0.35,

                "seasonality_strength": (0.35, 0.60),

                "monthly_growth_rate": (-0.02, 0.05)
            },

            "Trading of jewellery": {
                "weight": 0.40,

                "employee_range": {
                    "micro": (2, 10),
                    "small": (10, 35),
                    "medium": (35, 100)
                },

                "salary_range": (18000, 35000),

                "average_ticket_size": (3000, 300000),

                "b2b_share": (0.30, 0.60),

                "upi_share": (0.15, 0.35),
                "cash_share": (0.10, 0.25),
                "bank_transfer_share": (0.45, 0.70),

                "gst_rate": [3],

                "working_capital_days": (30, 75),

                "export_probability": 0.10,
                "e_invoice_probability": 0.25,

                "seasonality_strength": (0.40, 0.65),

                "monthly_growth_rate": (-0.02, 0.05)
            }
        }
    },
    "Green and Renewable Energy": {
        "category_weight": 0.01,

        "peak_months": [2, 3],
        "lean_months": [8],

        "peak_multiplier": (1.15, 1.35),
        "lean_multiplier": (0.85, 0.95),

        "subcategories": {

            "Solar panel manufacturing": {
                "weight": 0.40,

                "employee_range": {
                    "micro": (8, 25),
                    "small": (25, 100),
                    "medium": (100, 300)
                },

                "salary_range": (22000, 45000),

                "average_ticket_size": (50000, 1000000),

                "b2b_share": (0.90, 0.99),

                "upi_share": (0.00, 0.03),
                "cash_share": (0.00, 0.03),
                "bank_transfer_share": (0.94, 1.00),

                "gst_rate": [12, 18],

                "working_capital_days": (60, 120),

                "export_probability": 0.30,
                "e_invoice_probability": 0.75,

                "seasonality_strength": (0.10, 0.20),

                "monthly_growth_rate": (0.00, 0.08)
            },

            "Eco-friendly product production": {
                "weight": 0.60,

                "employee_range": {
                    "micro": (3, 15),
                    "small": (15, 60),
                    "medium": (60, 180)
                },

                "salary_range": (15000, 30000),

                "average_ticket_size": (500, 20000),

                "b2b_share": (0.45, 0.70),

                "upi_share": (0.20, 0.40),
                "cash_share": (0.10, 0.25),
                "bank_transfer_share": (0.40, 0.65),

                "gst_rate": [5, 12, 18],

                "working_capital_days": (30, 60),

                "export_probability": 0.20,
                "e_invoice_probability": 0.25,

                "seasonality_strength": (0.10, 0.25),

                "monthly_growth_rate": (0.01, 0.10)
            }
        }
    },
    "Information Technology Services": {
        "category_weight": 0.04,

        "peak_months": [],
        "lean_months": [],

        "peak_multiplier": (1.00, 1.10),
        "lean_multiplier": (0.90, 1.00),

        "subcategories": {

            "IT solutions providers": {
                "weight": 0.35,

                "employee_range": {
                    "micro": (3, 15),
                    "small": (15, 60),
                    "medium": (60, 200)
                },

                "salary_range": (25000, 70000),

                "average_ticket_size": (10000, 500000),

                "b2b_share": (0.80, 0.95),

                "upi_share": (0.01, 0.05),
                "cash_share": (0.00, 0.02),
                "bank_transfer_share": (0.93, 0.99),

                "gst_rate": [18],

                "working_capital_days": (15, 45),

                "export_probability": 0.20,
                "e_invoice_probability": 0.70,

                "seasonality_strength": (0.02, 0.08),

                "monthly_growth_rate": (0.00, 0.08)
            },

            "Software development firms": {
                "weight": 0.65,

                "employee_range": {
                    "micro": (5, 20),
                    "small": (20, 80),
                    "medium": (80, 300)
                },

                "salary_range": (30000, 90000),

                "average_ticket_size": (50000, 1000000),

                "b2b_share": (0.90, 0.99),

                "upi_share": (0.00, 0.03),
                "cash_share": (0.00, 0.01),
                "bank_transfer_share": (0.96, 1.00),

                "gst_rate": [18],

                "working_capital_days": (30, 90),

                "export_probability": 0.45,
                "e_invoice_probability": 0.85,

                "seasonality_strength": (0.02, 0.08),

                "monthly_growth_rate": (0.01, 0.10)
            }
        }
    },
    "Hospitality and Tourism": {
        "category_weight": 0.04,

        "peak_months": [10, 11, 12],
        "lean_months": [7, 8],

        "peak_multiplier": (1.25, 1.60),
        "lean_multiplier": (0.65, 0.85),

        "subcategories": {

            "Small hotels": {
                "weight": 0.45,

                "employee_range": {
                    "micro": (5, 20),
                    "small": (20, 80),
                    "medium": (80, 250)
                },

                "salary_range": (12000, 30000),

                "average_ticket_size": (1500, 20000),

                "b2b_share": (0.10, 0.30),
                "b2c_share": (0.70, 0.90),

                "upi_share": (0.35, 0.60),
                "cash_share": (0.10, 0.30),
                "bank_transfer_share": (0.20, 0.45),

                "gst_rate": [12, 18],

                "working_capital_days": (10, 30),

                "export_probability": 0.05,
                "e_invoice_probability": 0.15,

                "seasonality_strength": (0.35, 0.60),

                "monthly_growth_rate": (-0.03, 0.06)
            },

            "Guest houses": {
                "weight": 0.30,

                "employee_range": {
                    "micro": (3, 12),
                    "small": (12, 40),
                    "medium": (40, 120)
                },

                "salary_range": (12000, 25000),

                "average_ticket_size": (800, 8000),

                "b2b_share": (0.05, 0.20),
                "b2c_share": (0.80, 0.95),

                "upi_share": (0.45, 0.65),
                "cash_share": (0.15, 0.30),
                "bank_transfer_share": (0.10, 0.30),

                "gst_rate": [12],

                "working_capital_days": (7, 20),

                "export_probability": 0.02,
                "e_invoice_probability": 0.08,

                "seasonality_strength": (0.40, 0.65),

                "monthly_growth_rate": (-0.03, 0.05)
            },

            "Travel agencies": {
                "weight": 0.25,

                "employee_range": {
                    "micro": (2, 10),
                    "small": (10, 35),
                    "medium": (35, 120)
                },

                "salary_range": (18000, 40000),

                "average_ticket_size": (5000, 100000),

                "b2b_share": (0.30, 0.60),
                "b2c_share": (0.40, 0.70),

                "upi_share": (0.20, 0.40),
                "cash_share": (0.05, 0.15),
                "bank_transfer_share": (0.45, 0.65),

                "gst_rate": [18],

                "working_capital_days": (15, 45),

                "export_probability": 0.10,
                "e_invoice_probability": 0.25,

                "seasonality_strength": (0.35, 0.60),

                "monthly_growth_rate": (-0.02, 0.07)
            }
        }
    },
    "Healthcare Services": {
        "category_weight": 0.03,

        "peak_months": [7, 8],
        "lean_months": [2],

        "peak_multiplier": (1.15, 1.35),
        "lean_multiplier": (0.90, 1.00),

        "subcategories": {

            "Clinics": {
                "weight": 0.45,

                "employee_range": {
                    "micro": (3, 10),
                    "small": (10, 30),
                    "medium": (30, 80)
                },

                "salary_range": (18000, 45000),

                "average_ticket_size": (300, 3000),

                "b2b_share": (0.05, 0.15),
                "b2c_share": (0.85, 0.95),

                "upi_share": (0.45, 0.65),
                "cash_share": (0.10, 0.30),
                "bank_transfer_share": (0.15, 0.35),

                "gst_rate": [0],

                "working_capital_days": (5, 20),

                "export_probability": 0.00,
                "e_invoice_probability": 0.05,

                "seasonality_strength": (0.10, 0.20),

                "monthly_growth_rate": (-0.01, 0.05)
            },

            "Diagnostic centers": {
                "weight": 0.35,

                "employee_range": {
                    "micro": (5, 15),
                    "small": (15, 50),
                    "medium": (50, 150)
                },

                "salary_range": (22000, 50000),

                "average_ticket_size": (500, 8000),

                "b2b_share": (0.20, 0.40),
                "b2c_share": (0.60, 0.80),

                "upi_share": (0.35, 0.55),
                "cash_share": (0.10, 0.20),
                "bank_transfer_share": (0.25, 0.45),

                "gst_rate": [0, 18],

                "working_capital_days": (10, 30),

                "export_probability": 0.00,
                "e_invoice_probability": 0.10,

                "seasonality_strength": (0.10, 0.20),

                "monthly_growth_rate": (0.00, 0.05)
            },

            "Small hospitals": {
                "weight": 0.20,

                "employee_range": {
                    "micro": (15, 40),
                    "small": (40, 120),
                    "medium": (120, 350)
                },

                "salary_range": (22000, 60000),

                "average_ticket_size": (1000, 50000),

                "b2b_share": (0.20, 0.40),
                "b2c_share": (0.60, 0.80),

                "upi_share": (0.25, 0.45),
                "cash_share": (0.05, 0.15),
                "bank_transfer_share": (0.40, 0.60),

                "gst_rate": [0],

                "working_capital_days": (15, 45),

                "export_probability": 0.00,
                "e_invoice_probability": 0.15,

                "seasonality_strength": (0.08, 0.18),

                "monthly_growth_rate": (0.00, 0.06)
            }
        }
    }, 
    "Educational Services": {
        "category_weight": 0.04,

        "peak_months": [4, 5, 6],
        "lean_months": [12],

        "peak_multiplier": (1.20, 1.45),
        "lean_multiplier": (0.80, 0.95),

        "subcategories": {

            "Coaching centers": {
                "weight": 0.70,

                "employee_range": {
                    "micro": (3, 12),
                    "small": (12, 40),
                    "medium": (40, 120)
                },

                "salary_range": (18000, 50000),

                "average_ticket_size": (5000, 100000),

                "b2b_share": (0.05, 0.15),
                "b2c_share": (0.85, 0.95),

                "upi_share": (0.35, 0.60),
                "cash_share": (0.10, 0.30),
                "bank_transfer_share": (0.20, 0.45),

                "gst_rate": [18],

                "working_capital_days": (5, 20),

                "export_probability": 0.00,
                "e_invoice_probability": 0.08,

                "seasonality_strength": (0.30, 0.50),

                "monthly_growth_rate": (-0.02, 0.06)
            },

            "Vocational training institutes": {
                "weight": 0.30,

                "employee_range": {
                    "micro": (5, 15),
                    "small": (15, 50),
                    "medium": (50, 150)
                },

                "salary_range": (22000, 55000),

                "average_ticket_size": (10000, 150000),

                "b2b_share": (0.20, 0.50),
                "b2c_share": (0.50, 0.80),

                "upi_share": (0.20, 0.45),
                "cash_share": (0.05, 0.20),
                "bank_transfer_share": (0.35, 0.65),

                "gst_rate": [18],

                "working_capital_days": (10, 30),

                "export_probability": 0.05,
                "e_invoice_probability": 0.15,

                "seasonality_strength": (0.20, 0.40),

                "monthly_growth_rate": (-0.01, 0.08)
            }
        }
    },
    "Transportation Services": {
        "category_weight": 0.06,

        "peak_months": [10, 11, 12],
        "lean_months": [2],

        "peak_multiplier": (1.20, 1.50),
        "lean_multiplier": (0.80, 0.95),

        "subcategories": {

            "Logistics providers": {
                "weight": 0.65,

                "employee_range": {
                    "micro": (5, 20),
                    "small": (20, 80),
                    "medium": (80, 300)
                },

                "salary_range": (18000, 35000),

                "average_ticket_size": (5000, 200000),

                "b2b_share": (0.85, 0.98),
                "b2c_share": (0.02, 0.15),

                "upi_share": (0.02, 0.08),
                "cash_share": (0.02, 0.08),
                "bank_transfer_share": (0.85, 0.95),

                "gst_rate": [5, 12, 18],

                "working_capital_days": (30, 60),

                "export_probability": 0.12,
                "e_invoice_probability": 0.45,

                "seasonality_strength": (0.20, 0.35),

                "monthly_growth_rate": (-0.01, 0.05)
            },

            "Courier services": {
                "weight": 0.35,

                "employee_range": {
                    "micro": (3, 15),
                    "small": (15, 50),
                    "medium": (50, 150)
                },

                "salary_range": (16000, 30000),

                "average_ticket_size": (100, 3000),

                "b2b_share": (0.50, 0.75),
                "b2c_share": (0.25, 0.50),

                "upi_share": (0.20, 0.40),
                "cash_share": (0.10, 0.25),
                "bank_transfer_share": (0.40, 0.60),

                "gst_rate": [18],

                "working_capital_days": (10, 30),

                "export_probability": 0.02,
                "e_invoice_probability": 0.20,

                "seasonality_strength": (0.30, 0.45),

                "monthly_growth_rate": (0.00, 0.06)
            }
        }
    }, 
    "Retail Trade": {
        "category_weight": 0.12,

        "peak_months": [10, 11, 12],
        "lean_months": [6],

        "peak_multiplier": (1.30, 1.60),
        "lean_multiplier": (0.70, 0.90),

        "subcategories": {

            "Local shops": {
                "weight": 0.60,

                "employee_range": {
                    "micro": (2, 8),
                    "small": (8, 25),
                    "medium": (25, 80)
                },

                "salary_range": (12000, 22000),

                "average_ticket_size": (100, 3000),

                "b2b_share": (0.05, 0.20),
                "b2c_share": (0.80, 0.95),

                "upi_share": (0.35, 0.60),
                "cash_share": (0.25, 0.45),
                "bank_transfer_share": (0.05, 0.20),

                "gst_rate": [5, 12, 18],

                "working_capital_days": (15, 35),

                "export_probability": 0.00,
                "e_invoice_probability": 0.02,

                "seasonality_strength": (0.20, 0.40),

                "monthly_growth_rate": (-0.02, 0.05)
            },

            "Boutiques": {
                "weight": 0.10,

                "employee_range": {
                    "micro": (2, 8),
                    "small": (8, 20),
                    "medium": (20, 60)
                },

                "salary_range": (15000, 30000),

                "average_ticket_size": (500, 10000),

                "b2b_share": (0.05, 0.15),
                "b2c_share": (0.85, 0.95),

                "upi_share": (0.40, 0.60),
                "cash_share": (0.15, 0.30),
                "bank_transfer_share": (0.15, 0.30),

                "gst_rate": [5, 12],

                "working_capital_days": (20, 45),

                "export_probability": 0.02,
                "e_invoice_probability": 0.05,

                "seasonality_strength": (0.35, 0.55),

                "monthly_growth_rate": (-0.02, 0.06)
            },

            "Small retail outlets": {
                "weight": 0.30,

                "employee_range": {
                    "micro": (3, 12),
                    "small": (12, 40),
                    "medium": (40, 120)
                },

                "salary_range": (13000, 25000),

                "average_ticket_size": (200, 5000),

                "b2b_share": (0.10, 0.25),
                "b2c_share": (0.75, 0.90),

                "upi_share": (0.40, 0.65),
                "cash_share": (0.15, 0.35),
                "bank_transfer_share": (0.10, 0.25),

                "gst_rate": [5, 12, 18],

                "working_capital_days": (20, 40),

                "export_probability": 0.00,
                "e_invoice_probability": 0.08,

                "seasonality_strength": (0.25, 0.45),

                "monthly_growth_rate": (-0.02, 0.05)
            }
        }
    },
    "Real Estate and Renting Services": {
        "category_weight": 0.02,

        "peak_months": [2, 3],
        "lean_months": [8],

        "peak_multiplier": (1.15, 1.35),
        "lean_multiplier": (0.85, 0.95),

        "subcategories": {

            "Property management": {
                "weight": 0.45,

                "employee_range": {
                    "micro": (3, 12),
                    "small": (12, 40),
                    "medium": (40, 120)
                },

                "salary_range": (18000, 40000),

                "average_ticket_size": (5000, 100000),

                "b2b_share": (0.50, 0.70),
                "b2c_share": (0.30, 0.50),

                "upi_share": (0.10, 0.25),
                "cash_share": (0.05, 0.15),
                "bank_transfer_share": (0.65, 0.85),

                "gst_rate": [18],

                "working_capital_days": (15, 45),

                "export_probability": 0.00,
                "e_invoice_probability": 0.30,

                "seasonality_strength": (0.10, 0.20),

                "monthly_growth_rate": (-0.01, 0.04)
            },

            "Property rental services": {
                "weight": 0.55,

                "employee_range": {
                    "micro": (2, 8),
                    "small": (8, 25),
                    "medium": (25, 80)
                },

                "salary_range": (15000, 35000),

                "average_ticket_size": (10000, 200000),

                "b2b_share": (0.35, 0.60),
                "b2c_share": (0.40, 0.65),

                "upi_share": (0.10, 0.25),
                "cash_share": (0.02, 0.10),
                "bank_transfer_share": (0.70, 0.90),

                "gst_rate": [18],

                "working_capital_days": (5, 20),

                "export_probability": 0.00,
                "e_invoice_probability": 0.20,

                "seasonality_strength": (0.08, 0.18),

                "monthly_growth_rate": (-0.01, 0.04)
            }
        }
    },
    "Consultancy Services": {
        "category_weight": 0.04,

        "peak_months": [1, 4],
        "lean_months": [12],

        "peak_multiplier": (1.10, 1.30),
        "lean_multiplier": (0.85, 0.95),

        "subcategories": {

            "Business": {
                "weight": 0.45,

                "employee_range": {
                    "micro": (2, 8),
                    "small": (8, 30),
                    "medium": (30, 100)
                },

                "salary_range": (30000, 80000),

                "average_ticket_size": (10000, 500000),

                "b2b_share": (0.90, 0.98),
                "b2c_share": (0.02, 0.10),

                "upi_share": (0.02, 0.08),
                "cash_share": (0.00, 0.03),
                "bank_transfer_share": (0.90, 0.98),

                "gst_rate": [18],

                "working_capital_days": (15, 45),

                "export_probability": 0.15,
                "e_invoice_probability": 0.70,

                "seasonality_strength": (0.05, 0.15),

                "monthly_growth_rate": (0.00, 0.06)
            },

            "Financial": {
                "weight": 0.35,

                "employee_range": {
                    "micro": (2, 10),
                    "small": (10, 35),
                    "medium": (35, 120)
                },

                "salary_range": (35000, 90000),

                "average_ticket_size": (15000, 1000000),

                "b2b_share": (0.80, 0.95),
                "b2c_share": (0.05, 0.20),

                "upi_share": (0.01, 0.05),
                "cash_share": (0.00, 0.02),
                "bank_transfer_share": (0.93, 0.99),

                "gst_rate": [18],

                "working_capital_days": (10, 30),

                "export_probability": 0.10,
                "e_invoice_probability": 0.75,

                "seasonality_strength": (0.05, 0.12),

                "monthly_growth_rate": (0.00, 0.06)
            },

            "Legal consultancy firms": {
                "weight": 0.20,

                "employee_range": {
                    "micro": (2, 8),
                    "small": (8, 25),
                    "medium": (25, 80)
                },

                "salary_range": (35000, 100000),

                "average_ticket_size": (5000, 300000),

                "b2b_share": (0.60, 0.85),
                "b2c_share": (0.15, 0.40),

                "upi_share": (0.05, 0.15),
                "cash_share": (0.00, 0.05),
                "bank_transfer_share": (0.80, 0.92),

                "gst_rate": [18],

                "working_capital_days": (15, 45),

                "export_probability": 0.02,
                "e_invoice_probability": 0.60,

                "seasonality_strength": (0.05, 0.10),

                "monthly_growth_rate": (0.00, 0.05)
            }
        }
    },
    "Repair and Maintenance": {
        "category_weight": 0.05,

        "peak_months": [5, 6],
        "lean_months": [2],

        "peak_multiplier": (1.15, 1.35),
        "lean_multiplier": (0.85, 0.95),

        "subcategories": {

            "Vehicle workshop": {
                "weight": 0.55,

                "employee_range": {
                    "micro": (3, 12),
                    "small": (12, 40),
                    "medium": (40, 120)
                },

                "salary_range": (18000, 35000),

                "average_ticket_size": (500, 30000),

                "b2b_share": (0.30, 0.55),
                "b2c_share": (0.45, 0.70),

                "upi_share": (0.30, 0.50),
                "cash_share": (0.15, 0.30),
                "bank_transfer_share": (0.25, 0.45),

                "gst_rate": [18],

                "working_capital_days": (15, 45),

                "export_probability": 0.00,
                "e_invoice_probability": 0.10,

                "seasonality_strength": (0.10, 0.25),

                "monthly_growth_rate": (-0.01, 0.05)
            },

            "Appliance repairs workshop": {
                "weight": 0.45,

                "employee_range": {
                    "micro": (2, 10),
                    "small": (10, 30),
                    "medium": (30, 80)
                },

                "salary_range": (16000, 30000),

                "average_ticket_size": (300, 15000),

                "b2b_share": (0.20, 0.40),
                "b2c_share": (0.60, 0.80),

                "upi_share": (0.40, 0.60),
                "cash_share": (0.15, 0.30),
                "bank_transfer_share": (0.15, 0.35),

                "gst_rate": [18],

                "working_capital_days": (10, 30),

                "export_probability": 0.00,
                "e_invoice_probability": 0.05,

                "seasonality_strength": (0.20, 0.35),

                "monthly_growth_rate": (-0.01, 0.05)
            }
        }
    },
    "Creative Industries": {
        "category_weight": 0.03,

        "peak_months": [9, 10, 11],
        "lean_months": [1],

        "peak_multiplier": (1.20, 1.50),
        "lean_multiplier": (0.80, 0.95),

        "subcategories": {

            "Advertising agencies": {
                "weight": 0.40,

                "employee_range": {
                    "micro": (3, 12),
                    "small": (12, 40),
                    "medium": (40, 150)
                },

                "salary_range": (25000, 70000),

                "average_ticket_size": (10000, 500000),

                "b2b_share": (0.90, 0.98),
                "b2c_share": (0.02, 0.10),

                "upi_share": (0.02, 0.08),
                "cash_share": (0.00, 0.03),
                "bank_transfer_share": (0.90, 0.98),

                "gst_rate": [18],

                "working_capital_days": (30, 60),

                "export_probability": 0.10,
                "e_invoice_probability": 0.60,

                "seasonality_strength": (0.20, 0.40),

                "monthly_growth_rate": (0.00, 0.07)
            },

            "Design studios": {
                "weight": 0.35,

                "employee_range": {
                    "micro": (2, 10),
                    "small": (10, 35),
                    "medium": (35, 120)
                },

                "salary_range": (22000, 60000),

                "average_ticket_size": (5000, 300000),

                "b2b_share": (0.80, 0.95),
                "b2c_share": (0.05, 0.20),

                "upi_share": (0.05, 0.15),
                "cash_share": (0.00, 0.05),
                "bank_transfer_share": (0.80, 0.92),

                "gst_rate": [18],

                "working_capital_days": (20, 45),

                "export_probability": 0.20,
                "e_invoice_probability": 0.50,

                "seasonality_strength": (0.15, 0.30),

                "monthly_growth_rate": (0.01, 0.08)
            },

            "Media houses": {
                "weight": 0.25,

                "employee_range": {
                    "micro": (5, 20),
                    "small": (20, 70),
                    "medium": (70, 250)
                },

                "salary_range": (25000, 80000),

                "average_ticket_size": (20000, 1000000),

                "b2b_share": (0.85, 0.98),
                "b2c_share": (0.02, 0.15),

                "upi_share": (0.01, 0.05),
                "cash_share": (0.00, 0.02),
                "bank_transfer_share": (0.93, 0.99),

                "gst_rate": [18],

                "working_capital_days": (30, 75),

                "export_probability": 0.08,
                "e_invoice_probability": 0.70,

                "seasonality_strength": (0.20, 0.35),

                "monthly_growth_rate": (0.00, 0.06)
            }
        }
    },
    "Financial Services": {
        "category_weight": 0.02,

        "peak_months": [3, 4],
        "lean_months": [8],

        "peak_multiplier": (1.10, 1.25),
        "lean_multiplier": (0.90, 0.98),

        "subcategories": {

            "Microfinance institutions": {
                "weight": 0.70,

                "employee_range": {
                    "micro": (5, 20),
                    "small": (20, 80),
                    "medium": (80, 250)
                },

                "salary_range": (18000, 45000),

                "average_ticket_size": (5000, 100000),

                "b2b_share": (0.10, 0.30),
                "b2c_share": (0.70, 0.90),

                "upi_share": (0.20, 0.40),
                "cash_share": (0.10, 0.30),
                "bank_transfer_share": (0.40, 0.70),

                "gst_rate": [18],

                "working_capital_days": (15, 45),

                "export_probability": 0.00,
                "e_invoice_probability": 0.15,

                "seasonality_strength": (0.05, 0.15),

                "monthly_growth_rate": (0.00, 0.05)
            },

            "NBFCs": {
                "weight": 0.30,

                "employee_range": {
                    "micro": (10, 30),
                    "small": (30, 100),
                    "medium": (100, 350)
                },

                "salary_range": (30000, 80000),

                "average_ticket_size": (50000, 1000000),

                "b2b_share": (0.40, 0.70),
                "b2c_share": (0.30, 0.60),

                "upi_share": (0.05, 0.15),
                "cash_share": (0.00, 0.05),
                "bank_transfer_share": (0.85, 0.95),

                "gst_rate": [18],

                "working_capital_days": (10, 30),

                "export_probability": 0.02,
                "e_invoice_probability": 0.60,

                "seasonality_strength": (0.03, 0.10),

                "monthly_growth_rate": (0.00, 0.06)
            }
        }
    },
    "E-commerce and Digital Services": {
        "category_weight": 0.04,

        "peak_months": [10, 11],
        "lean_months": [2],

        "peak_multiplier": (1.25, 1.60),
        "lean_multiplier": (0.80, 0.95),

        "subcategories": {

            "Online retailers": {
                "weight": 0.65,

                "employee_range": {
                    "micro": (2, 10),
                    "small": (10, 40),
                    "medium": (40, 150)
                },

                "salary_range": (18000, 40000),

                "average_ticket_size": (300, 15000),

                "b2b_share": (0.10, 0.30),
                "b2c_share": (0.70, 0.90),

                "upi_share": (0.50, 0.70),
                "cash_share": (0.05, 0.15),
                "bank_transfer_share": (0.20, 0.40),

                "gst_rate": [5, 12, 18],

                "working_capital_days": (15, 45),

                "export_probability": 0.10,
                "e_invoice_probability": 0.20,

                "seasonality_strength": (0.35, 0.55),

                "monthly_growth_rate": (0.01, 0.08)
            },

            "Delivery services": {
                "weight": 0.35,

                "employee_range": {
                    "micro": (5, 20),
                    "small": (20, 80),
                    "medium": (80, 250)
                },

                "salary_range": (16000, 30000),

                "average_ticket_size": (50, 1000),

                "b2b_share": (0.60, 0.80),
                "b2c_share": (0.20, 0.40),

                "upi_share": (0.30, 0.55),
                "cash_share": (0.05, 0.20),
                "bank_transfer_share": (0.30, 0.50),

                "gst_rate": [18],

                "working_capital_days": (7, 20),

                "export_probability": 0.00,
                "e_invoice_probability": 0.15,

                "seasonality_strength": (0.25, 0.45),

                "monthly_growth_rate": (0.01, 0.07)
            }
        }
    },
    "Event Management": {
        "category_weight": 0.02,

        "peak_months": [10, 11, 12, 2],
        "lean_months": [6, 7],

        "peak_multiplier": (1.30, 1.70),
        "lean_multiplier": (0.60, 0.85),

        "subcategories": {

            "Event planning": {
                "weight": 0.70,

                "employee_range": {
                    "micro": (3, 12),
                    "small": (12, 40),
                    "medium": (40, 120)
                },

                "salary_range": (18000, 45000),

                "average_ticket_size": (10000, 1000000),

                "b2b_share": (0.40, 0.60),
                "b2c_share": (0.40, 0.60),

                "upi_share": (0.10, 0.25),
                "cash_share": (0.05, 0.15),
                "bank_transfer_share": (0.65, 0.85),

                "gst_rate": [18],

                "working_capital_days": (30, 75),

                "export_probability": 0.01,
                "e_invoice_probability": 0.30,

                "seasonality_strength": (0.40, 0.65),

                "monthly_growth_rate": (-0.02, 0.08)
            },

            "Exhibition management services": {
                "weight": 0.30,

                "employee_range": {
                    "micro": (5, 20),
                    "small": (20, 70),
                    "medium": (70, 200)
                },

                "salary_range": (22000, 50000),

                "average_ticket_size": (50000, 5000000),

                "b2b_share": (0.90, 0.98),
                "b2c_share": (0.02, 0.10),

                "upi_share": (0.02, 0.08),
                "cash_share": (0.00, 0.03),
                "bank_transfer_share": (0.90, 0.98),

                "gst_rate": [18],

                "working_capital_days": (45, 90),

                "export_probability": 0.05,
                "e_invoice_probability": 0.60,

                "seasonality_strength": (0.35, 0.55),

                "monthly_growth_rate": (-0.01, 0.06)
            }
        }
    },
    "Green and Renewable Energy Services": {
        "category_weight": 0.02,

        "peak_months": [2, 3],
        "lean_months": [8],

        "peak_multiplier": (1.15, 1.35),
        "lean_multiplier": (0.85, 0.95),

        "subcategories": {

            "Solar installations": {
                "weight": 0.60,

                "employee_range": {
                    "micro": (3, 12),
                    "small": (12, 40),
                    "medium": (40, 120)
                },

                "salary_range": (18000, 45000),

                "average_ticket_size": (50000, 1000000),

                "b2b_share": (0.40, 0.70),
                "b2c_share": (0.30, 0.60),

                "upi_share": (0.05, 0.15),
                "cash_share": (0.00, 0.05),
                "bank_transfer_share": (0.85, 0.95),

                "gst_rate": [12, 18],

                "working_capital_days": (30, 75),

                "export_probability": 0.05,
                "e_invoice_probability": 0.40,

                "seasonality_strength": (0.10, 0.25),

                "monthly_growth_rate": (0.01, 0.08)
            },

            "Waste management services": {
                "weight": 0.40,

                "employee_range": {
                    "micro": (5, 20),
                    "small": (20, 60),
                    "medium": (60, 180)
                },

                "salary_range": (15000, 35000),

                "average_ticket_size": (10000, 500000),

                "b2b_share": (0.70, 0.90),
                "b2c_share": (0.10, 0.30),

                "upi_share": (0.05, 0.15),
                "cash_share": (0.00, 0.05),
                "bank_transfer_share": (0.85, 0.95),

                "gst_rate": [18],

                "working_capital_days": (20, 60),

                "export_probability": 0.02,
                "e_invoice_probability": 0.35,

                "seasonality_strength": (0.05, 0.15),

                "monthly_growth_rate": (0.00, 0.05)
            }
        }
    }
}

'''
sectors = list(sector_config.keys())
weights = [sector_config[s]["category_weight"] for s in sectors]

print(weights)
print("Sum =", sum(weights))
print("Num sectors =", len(sectors))
print("Num weights =", len(weights))
'''
# ------------------------------------------------------------
# 1. Business-level sampling
# ------------------------------------------------------------
def generate_businesses(n=10000, seed=24):
    rng = np.random.default_rng(seed)
    businesses = []
    for _ in range(n):
        # MSME eligibility & bank relationships (as in your snippet)
        msme_eligible = rng.choice([True, False], p=[0.9, 0.1])
        has_our_bank = rng.choice([True, False], p=[0.3, 0.7])
        has_other_bank = rng.choice([True, False], p=[0.85, 0.15])

        if has_our_bank and has_other_bank:
            credit_hist = rng.choice(["None","Limited","Established"], p=[0.05,0.25,0.70])
        elif has_our_bank and not has_other_bank:
            credit_hist = rng.choice(["None","Limited","Established"], p=[0.20,0.40,0.40])
        elif not has_our_bank and has_other_bank:
            credit_hist = rng.choice(["None","Limited","Established"], p=[0.30,0.35,0.35])
        else:
            credit_hist = "None"

        if msme_eligible:
            size = rng.choice(["Micro","Small","Medium"], p=[0.55,0.35,0.10])
        '''
        else:
            size = rng.choice(["Large","Ineligible sector"], p=[0.80,0.20])
        '''

        # Business age (months)
        age_ranges = [(1,24),(25,60),(61,120),(121,180)]
        age_probs = [0.30,0.35,0.25,0.10]
        low, high = age_ranges[rng.choice(4, p=age_probs)]
        age_months = rng.integers(low, high+1)

        # Sector & subcategory using category_weight and subcategory weight
        sectors = list(sector_config.keys())
        weights = [sector_config[s]["category_weight"] for s in sectors]
        sector = rng.choice(sectors, p=weights)
        subcats = list(sector_config[sector]["subcategories"].keys())
        sub_weights = [sector_config[sector]["subcategories"][sc]["weight"] for sc in subcats]
        subcat = rng.choice(subcats, p=sub_weights)
 
        business = {
            "business_id": _,
            "msme_eligible": msme_eligible,
            "has_relationship_our_bank": has_our_bank,
            "has_relationship_other_bank": has_other_bank,
            "credit_history": credit_hist,
            "business_size": size,
            "sector": sector,
            "subcategory": subcat,
            "age_months": age_months,
        }
        businesses.append(business)
    return pd.DataFrame(businesses)

# ------------------------------------------------------------
# 2. Monthly time‑series generation for one business
# ------------------------------------------------------------
def generate_monthly_data(business, end_date=datetime(2026,7,15)):
    # Unpack config
    sec = business["sector"]
    sub = business["subcategory"]
    conf = sector_config[sec]
    subconf = conf["subcategories"][sub]
    size = business["business_size"]
    age = business["age_months"]

    rng = np.random.default_rng(business["business_id"] * 1000)  # reproducible per business

    reg_date = end_date - relativedelta(months=age)
    months = pd.date_range(reg_date, end_date, freq='MS')  # month starts
    n_months = len(months)

    # Base revenue drivers (scaled by business size)
    ticket_size_avg = rng.uniform(*subconf["average_ticket_size"])
    # Number of invoices per month – derived from employee count later
    emp_range = subconf["employee_range"][size.lower()]
    employees_base = rng.integers(*emp_range) / 2  # factor to avoid too many invoices
    # Base monthly revenue: employees * salary / some margin (approximation)
    # We'll set baseline revenue and simulate dynamics

    # Salary for cost simulation
    salary_avg = rng.uniform(*subconf["salary_range"])

    # Seasonality
    peak_months = conf["peak_months"]
    lean_months = conf["lean_months"]
    season_strength = rng.uniform(*subconf["seasonality_strength"])
    growth_rate = rng.uniform(*subconf["monthly_growth_rate"])

    # Payment shares
    upi_share = rng.uniform(*subconf["upi_share"])
    cash_share = rng.uniform(*subconf["cash_share"])
    bank_share = 1.0 - upi_share - cash_share

    # GST rate (random choice)
    gst_rate = rng.choice(subconf["gst_rate"]) / 100.0

    # Base revenue (annual) scaled by size category
    size_factor = {"Micro": 0.5, "Small": 1.0, "Medium": 2.5}.get(size, 1.0)
    annual_revenue_base = ticket_size_avg * employees_base * 12 * size_factor
    monthly_base = annual_revenue_base / 12

    # Data containers
    records = []
    # Bank account simulation
    balance = rng.uniform(10000, 100000)  # opening balance

    for i, month in enumerate(months):
        # Seasonality factor
        m = month.month
        if m in peak_months:
            mult = rng.uniform(*conf["peak_multiplier"])
        elif m in lean_months:
            mult = rng.uniform(*conf["lean_multiplier"])
        else:
            mult = 1.0
        # Trend
        trend = (1 + growth_rate) ** i
        noise = rng.normal(1, 0.05)  # 5% noise
        revenue = monthly_base * mult * trend * noise

        # GST
        taxable = revenue
        gst_amount = taxable * gst_rate

        # UPI
        upi_value = revenue * upi_share * rng.uniform(0.9, 1.1)
        upi_count = max(1, int(upi_value / rng.uniform(100, 500)))

        # Cash
        cash_value = revenue * cash_share * rng.uniform(0.9, 1.1)

        # Bank (AA) – credits from customers, debits for expenses
        bank_credit = revenue * bank_share * rng.uniform(0.95, 1.05)
        expenses = revenue * rng.uniform(0.4, 0.7)  # operating expenses
        salaries_paid = employees_base * salary_avg
        total_debits = expenses + salaries_paid
        balance = balance + bank_credit - total_debits
        balance = max(balance, 0)

        # EPFO: employees count stable with minor churn
        emp_count = max(1, int(employees_base * rng.uniform(0.95, 1.05)))
        pf_wages = salaries_paid * 0.12  # employer contribution
        epf_contribution = pf_wages

        record = {
            "month": month,
            "revenue": revenue,
            "taxable_turnover": taxable,
            "gst_amount": gst_amount,
            "upi_value": upi_value,
            "upi_count": upi_count,
            "cash_value": cash_value,
            "bank_credits": bank_credit,
            "bank_debits": total_debits,
            "bank_balance": balance,
            "employees": emp_count,
            "salary_paid": salaries_paid,
            "epfo_contribution": epf_contribution,
        }
        records.append(record)

    return pd.DataFrame(records)

def generate_gst_data(business, end_date=datetime(2026, 7, 15)):
    """GST filing data: turnover and tax liability."""
    monthly = generate_monthly_data(business, end_date=end_date)
    return monthly[["month", "revenue", "taxable_turnover", "gst_amount"]].copy()

def generate_upi_data(business, end_date=datetime(2026, 7, 15)):
    """UPI transaction data: digital payment value and volume."""
    monthly = generate_monthly_data(business, end_date=end_date)
    return monthly[["month", "upi_value", "upi_count"]].copy()

def generate_aa_data(business, end_date=datetime(2026, 7, 15)):
    """Account Aggregator (bank statement) data: credits, debits, balance, cash."""
    monthly = generate_monthly_data(business, end_date=end_date)
    return monthly[["month", "bank_credits", "bank_debits", "bank_balance", "cash_value"]].copy()

def generate_epfo_data(business, end_date=datetime(2026, 7, 15)):
    """EPFO data: employee headcount, salaries and employer PF contributions."""
    monthly = generate_monthly_data(business, end_date=end_date)
    return monthly[["month", "employees", "salary_paid", "epfo_contribution"]].copy()

# ------------------------------------------------------------
# 3. Feature engineering from monthly data
# ------------------------------------------------------------
def engineer_features(monthly_df, business):
    df = monthly_df.copy()
    # Revenue features
    rev = df["revenue"]
    rev_growth = rev.pct_change().mean()
    rev_std = rev.std()
    rev_mean = rev.mean()
    rev_cv = rev_std / (rev_mean + 1)
    # Seasonality strength: ratio of peak avg to off-peak avg (simplified)
    peak_months = sector_config[business["sector"]]["peak_months"]
    lean_months = sector_config[business["sector"]]["lean_months"]
    peak_avg = df[df["month"].dt.month.isin(peak_months)]["revenue"].mean()
    off_avg = df[df["month"].dt.month.isin(lean_months)]["revenue"].mean()
    seasonality_ratio = peak_avg / (off_avg + 1)

    # GST compliance: ratio of gst amount to taxable (should be constant ~rate)
    gst_rate_actual = (df["gst_amount"].sum() / df["taxable_turnover"].sum()) if df["taxable_turnover"].sum() > 0 else 0
    # UPI penetration
    upi_penetration = df["upi_value"].sum() / (df["revenue"].sum() + 1)
    # Cash flow consistency: bank balance CV
    bal_cv = df["bank_balance"].std() / (df["bank_balance"].mean() + 1)
    # Average EPFO contribution per employee
    avg_emp = df["employees"].mean()
    avg_salary = df["salary_paid"].mean() / (avg_emp + 1)
    pf_per_emp = df["epfo_contribution"].mean() / (avg_emp + 1)

    # Credit history numeric
    credit_map = {"None":0, "Limited":1, "Established":2}
    credit_score = credit_map[business["credit_history"]]

    features = {
        "business_id": business["business_id"],
        "sector": business["sector"],
        "subcategory": business["subcategory"],
        "size": business["business_size"],
        "age_months": business["age_months"],
        "credit_history": business["credit_history"],
        "has_relationship_our_bank": business["has_relationship_our_bank"],
        "has_relationship_other_bank": business["has_relationship_other_bank"],
        "revenue_mean": rev_mean,
        "revenue_growth": rev_growth,
        "revenue_cv": rev_cv,
        "seasonality_ratio": seasonality_ratio,
        "gst_effective_rate": gst_rate_actual,
        "upi_penetration": upi_penetration,
        "bank_balance_cv": bal_cv,
        "avg_employees": avg_emp,
        "avg_salary": avg_salary,
        "epfo_per_employee": pf_per_emp,
        "credit_history_score": credit_score,
    }
    return pd.Series(features)

# ------------------------------------------------------------
# 4. Target: Synthetic Financial Health Score (300–900)
# ------------------------------------------------------------
def compute_true_health_score(features):
    # Rules-based score, combining various dimensions
    # Revenue stability: lower CV is better
    rev_stab = max(0, 1 - features["revenue_cv"])
    # Growth: positive is good
    growth_norm = np.clip((features["revenue_growth"] + 0.1) * 5, 0, 1)
    # GST compliance: effective rate close to 0.05/0.12/0.18 is good; we assume actual == expected gives 1
    # Simplification: if between 0.04 and 0.20, considered compliant
    gst_ok = 1 if 0.01 < features["gst_effective_rate"] < 0.25 else 0
    # UPI penetration (proxy for digital footprint)
    upi_score = min(features["upi_penetration"] * 2, 1)  # higher is better up to 0.5
    # Cash flow health: low balance CV
    cash_health = max(0, 1 - features["bank_balance_cv"])
    # Credit history presence
    credit_factor = features["credit_history_score"] / 2.0  # 0-1
    # Business age maturity (6 years+)
    age_factor = min(features["age_months"] / 72, 1.0)
    # EPFO consistency (employees > 1, salary > some threshold)
    emp_factor = min(features["avg_employees"] / 10, 1.0)

    # Weighted sum
    score_raw = (0.15 * rev_stab +
                 0.10 * growth_norm +
                 0.15 * gst_ok +
                 0.10 * upi_score +
                 0.15 * cash_health +
                 0.15 * credit_factor +
                 0.10 * age_factor +
                 0.10 * emp_factor)
    # Scale to 300-900
    health_score = 300 + 600 * score_raw
    return health_score

# ------------------------------------------------------------
# 5. Main generation & modelling pipeline
# ------------------------------------------------------------
def main():
    # Generate 10,000 businesses
    print("Generating business master...")
    biz_df = generate_businesses(n=10000, seed=24)

    biz_df.to_csv("business_master.csv", index=False)
    print("Saved business_master.csv")

    all_monthly = []

    for idx, row in biz_df.iterrows():
        monthly = generate_monthly_data(row, end_date=datetime(2026, 7, 15))
        all_monthly.append(monthly)

    monthly_df = pd.concat(all_monthly, ignore_index=True)
    monthly_df.to_csv("monthly_transactions.csv", index=False)

    print("Saved monthly_transactions.csv")

    feature_list = []
    target_list = []

    for idx, row in biz_df.iterrows():

        monthly = generate_monthly_data(row, end_date=datetime(2026,7,15))

        features = engineer_features(monthly, row)
        score = compute_true_health_score(features)

        feature_list.append(features)
        target_list.append(score)

    features_df = pd.DataFrame(feature_list)
    features_df["true_health_score"] = target_list

    features_df.to_csv("engineered_features.csv", index=False)

    print("Saved engineered_features.csv")

    # For efficiency, sample a few to illustrate, but we'll process all
    n = len(biz_df)
    feature_list = []
    target_list = []

    for idx, row in biz_df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing business {idx}/{n}")
        monthly = generate_monthly_data(row, end_date=datetime(2026,7,15))
        features = engineer_features(monthly, row)
        true_score = compute_true_health_score(features)

        if pd.isna(true_score) or np.isinf(true_score):
            print(f"\nInvalid score at business {idx}")
            print(row)
            print(features)
            break

        feature_list.append(features)
        target_list.append(true_score)

    X = pd.DataFrame(feature_list)
    y = pd.Series(target_list, name="health_score")

    print("NaN labels:", y.isna().sum())
    print("Inf labels:", np.isinf(y).sum())
    print("Min:", y.min())
    print("Max:", y.max())

    if y.isna().any():
        print(y[y.isna()].head())

    # Drop non‑numeric identifiers for modelling
    X_model = X.drop(columns=["business_id","sector","subcategory","size",
                               "credit_history","has_relationship_our_bank",
                               "has_relationship_other_bank"])
    # Keep relationship flags for NTB/NTC analysis
    X_model["is_NTB"] = (~X["has_relationship_our_bank"]).astype(int)
    X_model["is_NTC"] = (X["credit_history"] == "None").astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y, test_size=0.2, random_state=42
    )

    # Model: XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"R²: {r2:.4f}, MAE: {mae:.2f}")

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=X_model.columns).sort_values(ascending=False)
    print("Top features:\n", importance.head(10))

    # Save model
    joblib.dump(model, "msme_health_score_model.pkl")

    # Example health card for a new business
    print("\n--- Example Health Card ---")
    new_biz = biz_df.iloc[0]  # take first synthetic business
    new_monthly = generate_monthly_data(new_biz)
    new_feats = engineer_features(new_monthly, new_biz)

    new_feats["is_NTB"] = int(not new_feats["has_relationship_our_bank"])
    new_feats["is_NTC"] = int(new_feats["credit_history"] == "None")

    #new_X = new_feats[X_model.columns]
    #new_X = pd.DataFrame([new_feats[X_model.columns]])
    new_X = pd.DataFrame([new_feats[X_model.columns]])
    new_X = new_X.astype(float)
    print(new_X.dtypes)
    #pred_score = model.predict(new_X.to_frame().T)[0]
    pred_score = model.predict(new_X)[0]
    print(f"Business ID {new_biz['business_id']}")
    print(f"Sector: {new_biz['sector']}, Subcategory: {new_biz['subcategory']}, Size: {new_biz['business_size']}")
    print(f"Predicted Financial Health Score: {pred_score:.0f} / 900")

if __name__ == "__main__":
    main()