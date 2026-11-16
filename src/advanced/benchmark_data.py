"""Advanced Challenge Benchmark Dataset for IPCC AR6 Synthesis Report (186 pages).
Contains 48 questions: 18 Dev (15 ans + 3 unans) and 30 Test (24 ans + 6 unans across all 6 required categories).
"""
from typing import Any

ADVANCED_BENCHMARK: list[dict[str, Any]] = [
    # =========================================================================
    # TẬP TEST: 30 CÂU (24 CÓ ĐÁP ÁN: 6 BẢNG, 5 SƠ ĐỒ, 5 BIỂU ĐỒ, 4 MULTI-HOP, 4 TEXT + 6 THIẾU BẰNG CHỨNG)
    # =========================================================================
    # --- NHÓM 1: BẢNG (TABLES) - 6 CÂU ---
    {
        "id": "test_tab_01",
        "split": "test",
        "main_group": "table",
        "question": "According to Table SPM.1 on page 37, by what percentage must global greenhouse gas (GHG) emissions be reduced by 2030, 2035, and 2040 compared to 2019 levels to limit warming to 1.5°C (>50%) with no or limited overshoot?",
        "answerable": True,
        "expected_answer": "43% [34-60%] by 2030, 60% [49-77%] by 2035, and 69% [58-90%] by 2040.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 37,
                "table_or_fig": "Table SPM.1",
                "bbox": [79, 78, 538, 149],
                "evidence": "Limit warming to1.5°C (>50%) with no or limited overshoot: GHG reductions by 2030: 43 [34-60]%; 2035: 60 [49-77]%; 2040: 69 [58-90]%.",
                "units": "% reduction relative to 2019",
            }
        ],
        "notes": "Tra chính xác hàng 1.5°C (>50%), dòng GHG, các cột năm 2030, 2035, 2040 trong Table SPM.1.",
    },
    {
        "id": "test_tab_02",
        "split": "test",
        "main_group": "table",
        "question": "Based on Table SPM.1 (page 37), what is the projected percentage reduction in CO2 emissions by 2050 required to limit warming to 1.5°C (>50%) versus to 2°C (>67%)?",
        "answerable": True,
        "expected_answer": "For 1.5°C (>50%), CO2 reduction by 2050 is 99% [79-119%] (net zero CO2); for 2°C (>67%), CO2 reduction by 2050 is 73% [59-90%].",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 37,
                "table_or_fig": "Table SPM.1",
                "bbox": [79, 78, 538, 149],
                "evidence": "1.5°C row CO2 2050: 99 [79-119]%; 2°C row CO2 2050: 73 [59-90]%.",
                "units": "% reduction relative to 2019",
            }
        ],
        "notes": "So sánh hai hàng kịch bản 1.5°C và 2°C tại cột 2050 đối với CO2.",
    },
    {
        "id": "test_tab_03",
        "split": "test",
        "main_group": "table",
        "question": "In Table SPM.1, for limiting global warming to 2°C (>67%), what are the median GHG emission reductions from 2019 levels specified for the years 2030 and 2050?",
        "answerable": True,
        "expected_answer": "21% [1-42%] reduction by 2030 and 64% [53-77%] reduction by 2050.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 37,
                "table_or_fig": "Table SPM.1",
                "bbox": [79, 78, 538, 149],
                "evidence": "Limit warming to 2°C (>67%): GHG 2030: 21 [1-42]%; GHG 2050: 64 [53-77]%.",
                "units": "%",
            }
        ],
        "notes": "Đọc hàng Limit warming to 2°C (>67%), dòng GHG, cột 2030 và 2050.",
    },
    {
        "id": "test_tab_04",
        "split": "test",
        "main_group": "table",
        "question": "According to Table 2.1 on page 61, what were the net anthropogenic greenhouse gas emissions in 2019 in GtCO2-eq, and what proportion came from CO2 from fossil fuel and industry?",
        "answerable": True,
        "expected_answer": "59 ± 6.6 GtCO2-eq in 2019, of which fossil fuel and industry CO2 accounted for approximately 64% (38 ± 3.0 GtCO2).",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 61,
                "table_or_fig": "Table 2.1",
                "bbox": [70, 100, 540, 260],
                "evidence": "Total net GHG emissions in 2019 were 59 ± 6.6 GtCO2-eq. CO2 from fossil fuel and industry was 38 ± 3.0 GtCO2 (approx 64%).",
                "units": "GtCO2-eq and %",
            }
        ],
        "notes": "Tra dữ liệu tổng phát thải 2019 và tỷ trọng phát thải hóa học/công nghiệp trong Table 2.1.",
    },
    {
        "id": "test_tab_05",
        "split": "test",
        "main_group": "table",
        "question": "From Table 3.1 on page 100, what is the estimated remaining carbon budget from the beginning of 2020 for a 50% likelihood of limiting warming to 1.5°C in GtCO2?",
        "answerable": True,
        "expected_answer": "Approximately 500 GtCO2 (or 400 GtCO2 for 67% likelihood).",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 100,
                "table_or_fig": "Table 3.1",
                "bbox": [75, 80, 540, 320],
                "evidence": "Remaining carbon budget from 2020: 500 GtCO2 for 50% probability of limiting warming to 1.5°C (400 GtCO2 for 67%).",
                "units": "GtCO2",
            }
        ],
        "notes": "Đọc bảng tổng hợp Remaining Carbon Budget ở Section 3 (Table 3.1 / Table SPM.2 context).",
    },
    {
        "id": "test_tab_06",
        "split": "test",
        "main_group": "table",
        "question": "According to Table 3.1 (page 100), in pathways that limit warming to 1.5°C (>50%) with no or limited overshoot, in what approximate year is global net zero CO2 reached?",
        "answerable": True,
        "expected_answer": "In the early 2050s (around 2050-2055; median around 2050).",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 100,
                "table_or_fig": "Table 3.1",
                "bbox": [75, 80, 540, 320],
                "evidence": "Net zero CO2 timing: median year is around 2050 (early 2050s [2035-2070]) for category C1 (1.5°C with no or limited overshoot).",
                "units": "Year",
            }
        ],
        "notes": "Tra mốc thời gian net zero CO2 theo kịch bản C1 trong Table 3.1.",
    },

    # --- NHÓM 2: ẢNH, SƠ ĐỒ (DIAGRAMS) - 5 CÂU ---
    {
        "id": "test_diag_01",
        "split": "test",
        "main_group": "diagram",
        "question": "In Figure SPM.1 panel (a) on page 23, what are the three major domains of human systems and ecosystems evaluated for adverse impacts and losses attributed to climate change?",
        "answerable": True,
        "expected_answer": "Water availability and food production, Health and well-being, and Cities, settlements and infrastructure, alongside ecosystem domains (terrestrial, freshwater, and ocean).",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 23,
                "table_or_fig": "Figure SPM.1(a)",
                "bbox": [70, 75, 540, 350],
                "evidence": "Figure SPM.1(a) assesses impacts across: Water availability & food production; Health & well-being; and Cities, settlements & infrastructure, as well as Terrestrial, freshwater and ocean ecosystems.",
                "units": "System domains",
            }
        ],
        "notes": "Đọc các nhóm cột lớn và nhãn phân loại hệ thống trong sơ đồ Figure SPM.1(a).",
    },
    {
        "id": "test_diag_02",
        "split": "test",
        "main_group": "diagram",
        "question": "Looking at Figure SPM.1 panel (c) on page 23, how does the diagram illustrate the historical and projected warming experience across different human generations born in 1950, 1980, and 2020?",
        "answerable": True,
        "expected_answer": "It shows that a person born in 1950 experienced modest warming during their life, someone born in 1980 experiences notable warming, while a person born in 2020 will face dramatically higher temperatures under higher emission scenarios as they reach age 70.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 23,
                "table_or_fig": "Figure SPM.1(c)",
                "bbox": [70, 480, 540, 720],
                "evidence": "Figure SPM.1(c) illustrates how climate has already changed and will change along lifespan bars for people born in 1950, 1980 and 2020 under different SSP scenarios.",
                "units": "Lifespan warming trajectories",
            }
        ],
        "notes": "Mô tả sơ đồ các dải màu thế hệ người sinh năm 1950, 1980, 2020 trong Figure SPM.1(c).",
    },
    {
        "id": "test_diag_03",
        "split": "test",
        "main_group": "diagram",
        "question": "In Figure SPM.2 on page 28, what are the three natural carbon sink reservoirs depicted, and what happens to their absorption efficiency as cumulative CO2 emissions increase?",
        "answerable": True,
        "expected_answer": "The atmosphere, land sink, and ocean sink. As cumulative CO2 emissions increase, the relative proportion absorbed by land and ocean sinks decreases (sinks become less effective).",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 28,
                "table_or_fig": "Figure SPM.2",
                "bbox": [70, 75, 540, 400],
                "evidence": "Figure SPM.2 shows partition into atmosphere, land and ocean. Sinks take up a smaller fraction of emissions in scenarios with higher cumulative emissions.",
                "units": "Fraction of cumulative emissions",
            }
        ],
        "notes": "Sơ đồ bồn hấp thụ carbon tự nhiên (đất, đại dương, khí quyển) trong Figure SPM.2.",
    },
    {
        "id": "test_diag_04",
        "split": "test",
        "main_group": "diagram",
        "question": "Examining Figure SPM.6 on page 41, what concept does the diagram illustrate using diverging colored pathways (ranging from red to green) leading to 2100?",
        "answerable": True,
        "expected_answer": "Climate resilient development pathways, showing how immediate societal choices and actions can transition systems towards higher sustainability and resilience (green) rather than higher climate risk (red).",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 41,
                "table_or_fig": "Figure SPM.6",
                "bbox": [70, 75, 540, 450],
                "evidence": "Figure SPM.6: Illustrative development pathways (red to green) and associated outcomes, showing the narrowing window of opportunity for Climate Resilient Development.",
                "units": "Development pathways",
            }
        ],
        "notes": "Sơ đồ nhánh rẽ các con đường phát triển thích ứng khí hậu (Figure SPM.6).",
    },
    {
        "id": "test_diag_05",
        "split": "test",
        "main_group": "diagram",
        "question": "In the schematic diagram Figure 1.1 on page 54, what are the components of climate risk depicted in the intersecting circles framework?",
        "answerable": True,
        "expected_answer": "Vulnerability, Exposure, and Hazard, which interact under the influence of climate change and human society.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 54,
                "table_or_fig": "Figure 1.1",
                "bbox": [70, 100, 540, 380],
                "evidence": "Figure 1.1 illustrates risk as the intersection of Hazard, Exposure, and Vulnerability within coupled human and natural systems.",
                "units": "Conceptual framework",
            }
        ],
        "notes": "Sơ đồ khái niệm rủi ro khí hậu (Hazard, Vulnerability, Exposure) ở trang 54.",
    },

    # --- NHÓM 3: BIỂU ĐỒ (CHARTS) - 5 CÂU ---
    {
        "id": "test_chart_01",
        "split": "test",
        "main_group": "chart",
        "question": "In Figure SPM.4 panel (a) on page 34, what are the five illustrative Reasons for Concern (RFCs) evaluated across global warming levels from 0°C to 5°C?",
        "answerable": True,
        "expected_answer": "RFC1: Unique and threatened systems; RFC2: Extreme weather events; RFC3: Distribution of impacts; RFC4: Global aggregate impacts; RFC5: Large-scale singular events.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 34,
                "table_or_fig": "Figure SPM.4(a)",
                "bbox": [70, 75, 540, 400],
                "evidence": "Figure SPM.4(a) burning embers show RFC1 (Unique and threatened systems), RFC2 (Extreme weather events), RFC3 (Distribution of impacts), RFC4 (Global aggregate impacts), RFC5 (Large-scale singular events).",
                "units": "Risk categories (RFC1 to RFC5)",
            }
        ],
        "notes": "Biểu đồ cột than hồng (burning embers) về 5 nhóm Reasons for Concern.",
    },
    {
        "id": "test_chart_02",
        "split": "test",
        "main_group": "chart",
        "question": "According to the bar chart in Figure SPM.7 panel (a) on page 43, which two specific energy mitigation options have the highest potential for net emission reduction by 2030 at costs below 20 USD per tCO2-eq?",
        "answerable": True,
        "expected_answer": "Solar energy and Wind energy.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 43,
                "table_or_fig": "Figure SPM.7(a)",
                "bbox": [70, 75, 540, 500],
                "evidence": "Figure SPM.7(a) Energy sector: Solar energy and Wind energy have the longest blue/light-blue bars (cost < 0 and 0-20 USD/tCO2-eq), contributing several GtCO2-eq/yr each.",
                "units": "Mitigation options (Energy)",
            }
        ],
        "notes": "Đọc biểu đồ thanh ngang Figure SPM.7(a) lĩnh vực Năng lượng (Energy).",
    },
    {
        "id": "test_chart_03",
        "split": "test",
        "main_group": "chart",
        "question": "From the chart in Figure SPM.5 panel (a) on page 39, what was the cumulative net historical CO2 emission from 1850 to 2019 in GtCO2?",
        "answerable": True,
        "expected_answer": "Approximately 2400 ± 240 GtCO2.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 39,
                "table_or_fig": "Figure SPM.5(a)",
                "bbox": [70, 75, 540, 350],
                "evidence": "Figure SPM.5(a): Cumulative CO2 emissions between 1850 and 2019 were 2400 ± 240 GtCO2.",
                "units": "GtCO2",
            }
        ],
        "notes": "Đọc trục phát thải CO2 tích lũy lịch sử 1850–2019 trong Figure SPM.5(a).",
    },
    {
        "id": "test_chart_04",
        "split": "test",
        "main_group": "chart",
        "question": "In Figure SPM.7 panel (a) under the 'Agriculture, forestry and other land use' (AFOLU) sector on page 43, which option provides the largest overall mitigation potential?",
        "answerable": True,
        "expected_answer": "Reduction of conversion of forests and other ecosystems (reduced deforestation).",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 43,
                "table_or_fig": "Figure SPM.7(a)",
                "bbox": [70, 250, 540, 600],
                "evidence": "Under AFOLU in Figure SPM.7(a), 'Reduce conversion of forests and other ecosystems' shows the largest total mitigation potential bar.",
                "units": "GtCO2-eq/yr",
            }
        ],
        "notes": "Đọc các tùy chọn giảm phát thải trong nhóm AFOLU tại Figure SPM.7(a).",
    },
    {
        "id": "test_chart_05",
        "split": "test",
        "main_group": "chart",
        "question": "Based on the warming curves in Figure 2.1 on page 58, what is the assessed range of global surface temperature increase in 2011-2020 compared to the 1850-1900 baseline?",
        "answerable": True,
        "expected_answer": "1.09°C [0.95°C to 1.20°C] higher.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 58,
                "table_or_fig": "Figure 2.1",
                "bbox": [70, 75, 540, 380],
                "evidence": "Figure 2.1 and accompanying text: Global surface temperature was 1.09 [0.95 to 1.20] °C higher in 2011–2020 than in 1850–1900.",
                "units": "°C",
            }
        ],
        "notes": "Đọc giá trị nhiệt độ bề mặt toàn cầu giai đoạn 2011–2020 trong Figure 2.1.",
    },

    # --- NHÓM 4: MULTI-HOP (LIÊN KẾT NHIỀU TRANG/BẢNG/HÌNH) - 4 CÂU ---
    {
        "id": "test_mhop_01",
        "split": "test",
        "main_group": "multi_hop",
        "question": "Combine information from Table SPM.1 (page 37) and Section 3 text on page 98: what is the 2030 GHG reduction required for 1.5°C (>50%), and how does the remaining carbon budget from 2020 (500 GtCO2) compare to the projected cumulative emissions until 2030 under current policies?",
        "answerable": True,
        "expected_answer": "Table SPM.1 requires a 43% [34-60%] GHG reduction by 2030; page 98 indicates that projected cumulative CO2 emissions from existing and planned fossil fuel infrastructure would exceed the remaining carbon budget of 500 GtCO2 for 1.5°C.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 37,
                "table_or_fig": "Table SPM.1",
                "bbox": [79, 78, 538, 149],
                "evidence": "Table SPM.1: 43% [34-60%] GHG reduction by 2030 for 1.5°C (>50%).",
                "units": "%",
            },
            {
                "source_id": "ipcc-ar6-syr",
                "page": 98,
                "table_or_fig": "Section 3.3.1 text",
                "bbox": [70, 100, 540, 450],
                "evidence": "Page 98: Cumulative future CO2 emissions from existing fossil fuel infrastructure alone exceed the remaining carbon budget for 1.5°C (500 GtCO2).",
                "units": "GtCO2 comparison",
            },
        ],
        "notes": "Kết hợp Table SPM.1 ở trang 37 và văn bản phân tích ngân sách carbon ở trang 98.",
    },
    {
        "id": "test_mhop_02",
        "split": "test",
        "main_group": "multi_hop",
        "question": "Cross-referencing Figure SPM.1(b) on page 23 and Figure SPM.3 on page 30: what key weather extreme is shown to have increased attributable to human influence across most land regions, and at what warming levels (1.5°C vs 2°C) do associated heat extremes reach high risk?",
        "answerable": True,
        "expected_answer": "Hot extremes / extreme heat has increased with high confidence across most land regions; associated risks transition to high risk between 1.5°C and 2°C of global warming.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 23,
                "table_or_fig": "Figure SPM.1(b)",
                "bbox": [70, 250, 540, 500],
                "evidence": "Figure SPM.1(b): Human influence has contributed to observed increases in hot extremes across virtually all inhabited regions.",
                "units": "Observed attribution",
            },
            {
                "source_id": "ipcc-ar6-syr",
                "page": 30,
                "table_or_fig": "Figure SPM.3",
                "bbox": [70, 75, 540, 400],
                "evidence": "Figure SPM.3: Heat-related human health risks and extreme heat events reach high risk at 1.5°C to 2°C warming.",
                "units": "Projected risk levels",
            },
        ],
        "notes": "Kết hợp sơ đồ bằng chứng lịch sử (trang 23) và biểu đồ nguy cơ tương lai (trang 30).",
    },
    {
        "id": "test_mhop_03",
        "split": "test",
        "main_group": "multi_hop",
        "question": "Synthesizing Figure SPM.7(a) (page 43) and Table SPM.1 (page 37): how do the low-cost mitigation potentials of wind and solar energy by 2030 support the required 48% CO2 emission reduction for 1.5°C pathways?",
        "answerable": True,
        "expected_answer": "Figure SPM.7(a) shows that solar and wind together can provide over 8 GtCO2-eq/yr of reductions at low/negative costs, which constitutes a substantial fraction of the global 48% CO2 reduction needed by 2030 shown in Table SPM.1.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 43,
                "table_or_fig": "Figure SPM.7(a)",
                "bbox": [70, 75, 540, 450],
                "evidence": "Solar and wind each offer 3-5 GtCO2-eq/yr at costs < 20 USD/tCO2-eq.",
                "units": "GtCO2-eq/yr",
            },
            {
                "source_id": "ipcc-ar6-syr",
                "page": 37,
                "table_or_fig": "Table SPM.1",
                "bbox": [79, 78, 538, 149],
                "evidence": "Table SPM.1: CO2 reduction of 48% [36-69%] required by 2030.",
                "units": "%",
            },
        ],
        "notes": "Kết hợp tiềm năng năng lượng tái tạo (trang 43) với mục tiêu giảm phát thải CO2 2030 (trang 37).",
    },
    {
        "id": "test_mhop_04",
        "split": "test",
        "main_group": "multi_hop",
        "question": "Combining Section 2.1 (page 57) on observed impacts and Section 4.3 (page 115) on urban areas: what percentage of the global population is estimated to live in highly vulnerable contexts, and what urban adaptation strategies are highlighted to reduce their vulnerability?",
        "answerable": True,
        "expected_answer": "Approximately 3.3 to 3.6 billion people live in contexts that are highly vulnerable to climate change; key urban strategies include green and blue infrastructure, climate-resilient water systems, and land-use planning.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 57,
                "table_or_fig": "Section 2.1.2 text",
                "bbox": [70, 100, 540, 300],
                "evidence": "Approximately 3.3 to 3.6 billion people live in contexts that are highly vulnerable to climate change.",
                "units": "Billion people",
            },
            {
                "source_id": "ipcc-ar6-syr",
                "page": 115,
                "table_or_fig": "Section 4.3 text",
                "bbox": [70, 100, 540, 350],
                "evidence": "Urban adaptation combines green/blue infrastructure, drainage, sustainable building codes and land use zoning.",
                "units": "Adaptation options",
            },
        ],
        "notes": "Kết hợp số liệu dân số dễ tổn thương (trang 57) và giải pháp đô thị (trang 115).",
    },

    # --- NHÓM 5: VĂN BẢN (TEXT) - 4 CÂU ---
    {
        "id": "test_text_01",
        "split": "test",
        "main_group": "text",
        "question": "According to the SPM text on page 19 (paragraph A.1), what is the unambiguous scientific consensus regarding human influence on global surface temperature warming?",
        "answerable": True,
        "expected_answer": "Human activities, principally through emissions of greenhouse gases, have unequivocally caused global warming, with global surface temperature reaching 1.1°C above 1850-1900 in 2011-2020.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 19,
                "table_or_fig": "SPM A.1 text",
                "bbox": [70, 100, 540, 250],
                "evidence": "Human activities, principally through emissions of greenhouse gases, have unequivocally caused global warming, with global surface temperature reaching 1.1°C above 1850–1900 in 2011–2020.",
                "units": "Paragraph quote",
            }
        ],
        "notes": "Trích xuất câu kết luận cốt lõi ở đoạn A.1 trang 19.",
    },
    {
        "id": "test_text_02",
        "split": "test",
        "main_group": "text",
        "question": "In paragraph B.1.1 on page 27, what is stated about when global warming is projected to exceed 1.5°C under the assessed emissions scenarios?",
        "answerable": True,
        "expected_answer": "Global warming is more likely than not to reach 1.5°C in the near term (2021-2040) even under the very low GHG emissions scenario, and will exceed 1.5°C under intermediate, high, and very high emissions scenarios.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 27,
                "table_or_fig": "SPM B.1.1 text",
                "bbox": [70, 120, 540, 300],
                "evidence": "Global warming is more likely than not to reach 1.5°C in the near term (2021–2040) under very low GHG emissions scenarios and is projected to exceed 1.5°C under intermediate, high and very high scenarios.",
                "units": "Scenario comparison",
            }
        ],
        "notes": "Đọc điều kiện vượt ngưỡng 1.5°C trong đoạn B.1.1 trang 27.",
    },
    {
        "id": "test_text_03",
        "split": "test",
        "main_group": "text",
        "question": "According to Section 4.2 on page 112, what is the 'adaptation gap' and why does current adaptation finance fall short?",
        "answerable": True,
        "expected_answer": "Current global financial flows for adaptation are insufficient for, and constrain implementation of, adaptation options, especially in developing countries, creating an adaptation gap between implemented action and societal needs.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 112,
                "table_or_fig": "Section 4.2 text",
                "bbox": [70, 100, 540, 320],
                "evidence": "Current global financial flows for adaptation are insufficient for, and constrain implementation of, adaptation options, especially in developing countries.",
                "units": "Policy finding",
            }
        ],
        "notes": "Trích đoạn về khoảng trống tài chính thích ứng ở trang 112.",
    },
    {
        "id": "test_text_04",
        "split": "test",
        "main_group": "text",
        "question": "In paragraph C.1 on page 36, what is stated regarding the effectiveness of adaptation options as global warming increases?",
        "answerable": True,
        "expected_answer": "Adaptation options that are feasible and effective today will become constrained and less effective with increasing global warming.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 36,
                "table_or_fig": "SPM C.1 text",
                "bbox": [70, 100, 540, 260],
                "evidence": "Adaptation options that are feasible and effective today will become constrained and less effective with increasing global warming.",
                "units": "Scientific statement",
            }
        ],
        "notes": "Trích dẫn nguyên văn khẳng định về tính hiệu quả của các giải pháp thích ứng tại đoạn C.1 trang 36.",
    },

    # --- NHÓM 6: THIẾU BẰNG CHỨNG (UNANSWERABLE) - 6 CÂU ---
    {
        "id": "test_unans_01",
        "split": "test",
        "main_group": "unanswerable",
        "question": "What is the specific municipal property tax rate in percent recommended by the IPCC AR6 Synthesis Report for the city of Zurich to finance flood protection dykes?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Searched all 186 pages for 'Zurich' and 'property tax'. The IPCC AR6 Synthesis Report provides global and regional assessments and does not recommend specific municipal tax rates for individual cities like Zurich.",
    },
    {
        "id": "test_unans_02",
        "split": "test",
        "main_group": "unanswerable",
        "question": "According to Table SPM.1, what is the exact chemical composition percentage of ruthenium catalyst used in direct air carbon capture facilities in 2030?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Checked Table SPM.1 (page 37) and entire document for 'ruthenium'. Table SPM.1 only reports GHG and CO2 percentage emission reduction trajectories; specific chemical catalyst compositions do not exist in the report.",
    },
    {
        "id": "test_unans_03",
        "split": "test",
        "main_group": "unanswerable",
        "question": "In Figure SPM.7, what is the exact dollar price per kilowatt-hour of household residential electricity tariff in Tokyo under the 2°C pathway?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Checked Figure SPM.7 (pages 43-46) and full text. Figure SPM.7 presents mitigation potential in GtCO2-eq/yr categorized by cost ranges in USD/tCO2-eq, not retail residential electricity tariffs for specific cities.",
    },
    {
        "id": "test_unans_04",
        "split": "test",
        "main_group": "unanswerable",
        "question": "What is the personal mobile phone number and private home address of IPCC Chair Hoesung Lee listed in Section 1 of the report?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Checked Section 1 and annexes. Hoesung Lee is credited as IPCC Chair, but personal contact information (mobile numbers, home addresses) is strictly excluded from scientific publications.",
    },
    {
        "id": "test_unans_05",
        "split": "test",
        "main_group": "unanswerable",
        "question": "According to the Synthesis Report, which specific national football team won the FIFA World Cup during the year of the AR6 synthesis approval?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Searched full text for 'FIFA', 'World Cup', 'football'. The document strictly evaluates climate science, vulnerability, and mitigation, containing zero sports tournament information.",
    },
    {
        "id": "test_unans_06",
        "split": "test",
        "main_group": "unanswerable",
        "question": "What is the projected gross domestic product (GDP) in billions of USD for the Republic of Atlantis in the year 2080 under scenario SSP5-8.5?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Searched all 186 pages for 'Atlantis'. 'Atlantis' is a fictional/mythological place and does not exist in real-world IPCC socioeconomic pathways.",
    },

    # =========================================================================
    # TẬP DEV: 18 CÂU (15 CÓ ĐÁP ÁN: 3 BẢNG, 3 SƠ ĐỒ, 3 BIỂU ĐỒ, 3 MULTI-HOP, 3 TEXT + 3 THIẾU BẰNG CHỨNG)
    # =========================================================================
    {
        "id": "dev_tab_01",
        "split": "dev",
        "main_group": "table",
        "question": "According to Table SPM.1 on page 37, what is the required CO2 reduction percentage from 2019 levels by 2030 to limit warming to 1.5°C (>50%)?",
        "answerable": True,
        "expected_answer": "48% [36-69%].",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 37,
                "table_or_fig": "Table SPM.1",
                "bbox": [79, 78, 538, 149],
                "evidence": "Limit warming to 1.5°C (>50%): CO2 2030: 48 [36-69]%.",
                "units": "%",
            }
        ],
    },
    {
        "id": "dev_tab_02",
        "split": "dev",
        "main_group": "table",
        "question": "In Table SPM.1, by how much must greenhouse gases (GHG) be reduced by 2050 to limit warming to 2°C (>67%)?",
        "answerable": True,
        "expected_answer": "64% [53-77%].",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 37,
                "table_or_fig": "Table SPM.1",
                "bbox": [79, 78, 538, 149],
                "evidence": "Limit warming to 2°C (>67%): GHG 2050: 64 [53-77]%.",
                "units": "%",
            }
        ],
    },
    {
        "id": "dev_tab_03",
        "split": "dev",
        "main_group": "table",
        "question": "According to Table 2.1 on page 61, what was the estimated global net anthropogenic GHG emission in 2019?",
        "answerable": True,
        "expected_answer": "59 ± 6.6 GtCO2-eq.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 61,
                "table_or_fig": "Table 2.1",
                "bbox": [70, 100, 540, 260],
                "evidence": "Total net GHG emissions in 2019: 59 ± 6.6 GtCO2-eq.",
                "units": "GtCO2-eq",
            }
        ],
    },
    {
        "id": "dev_diag_01",
        "split": "dev",
        "main_group": "diagram",
        "question": "In Figure SPM.1(c) on page 23, what does the warming gradient from blue to purple illustrate regarding historical and future climate?",
        "answerable": True,
        "expected_answer": "It illustrates the progression from historical cooler climate (blue) through recent warming (yellow/red) to much hotter future conditions (purple/dark red) under high emission scenarios.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 23,
                "table_or_fig": "Figure SPM.1(c)",
                "bbox": [70, 480, 540, 720],
                "evidence": "Figure SPM.1(c) color bar represents global surface temperature change relative to 1850-1900.",
                "units": "Temperature anomaly color scale",
            }
        ],
    },
    {
        "id": "dev_diag_02",
        "split": "dev",
        "main_group": "diagram",
        "question": "In Figure SPM.2 on page 28, what does the decreasing percentage of CO2 taken up by land and ocean sinks indicate?",
        "answerable": True,
        "expected_answer": "It indicates that natural carbon sinks become less effective as cumulative CO2 emissions increase.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 28,
                "table_or_fig": "Figure SPM.2",
                "bbox": [70, 75, 540, 400],
                "evidence": "Natural land and ocean carbon sinks take up a decreasing proportion of emissions as cumulative emissions increase.",
                "units": "Sink efficiency",
            }
        ],
    },
    {
        "id": "dev_diag_03",
        "split": "dev",
        "main_group": "diagram",
        "question": "What are the two major axes of risk shown in the conceptual diagram Figure 1.1 on page 54?",
        "answerable": True,
        "expected_answer": "Climate system hazards on one hand, and human society exposure and vulnerability on the other.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 54,
                "table_or_fig": "Figure 1.1",
                "bbox": [70, 100, 540, 380],
                "evidence": "Interaction between climate hazards and vulnerability/exposure of human and natural systems.",
                "units": "Risk framework",
            }
        ],
    },
    {
        "id": "dev_chart_01",
        "split": "dev",
        "main_group": "chart",
        "question": "In Figure SPM.4(a) on page 34, which Reason for Concern (RFC) exhibits very high risk at the lowest global warming level (around 1.5°C)?",
        "answerable": True,
        "expected_answer": "RFC1: Unique and threatened systems.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 34,
                "table_or_fig": "Figure SPM.4(a)",
                "bbox": [70, 75, 540, 400],
                "evidence": "RFC1 (Unique and threatened systems) transitions to very high risk (purple) near 1.5°C.",
                "units": "RFC category",
            }
        ],
    },
    {
        "id": "dev_chart_02",
        "split": "dev",
        "main_group": "chart",
        "question": "According to Figure SPM.7(a) on page 43, what is the color code indicating mitigation options whose costs are lower than the reference energy system?",
        "answerable": True,
        "expected_answer": "Blue (indicating costs less than 0 USD per tCO2-eq / net cost-saving).",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 43,
                "table_or_fig": "Figure SPM.7(a)",
                "bbox": [70, 75, 540, 500],
                "evidence": "Legend in Figure SPM.7: Blue bar indicates 'Costs are lower than the reference' (< 0 USD/tCO2-eq).",
                "units": "Legend color",
            }
        ],
    },
    {
        "id": "dev_chart_03",
        "split": "dev",
        "main_group": "chart",
        "question": "In Figure SPM.5(a) on page 39, what proportion of total historical cumulative CO2 emissions (1850-2019) occurred between 1990 and 2019?",
        "answerable": True,
        "expected_answer": "Approximately 42% of cumulative emissions occurred between 1990 and 2019.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 39,
                "table_or_fig": "Figure SPM.5(a)",
                "bbox": [70, 75, 540, 350],
                "evidence": "42% of historical cumulative net CO2 emissions occurred between 1990 and 2019.",
                "units": "%",
            }
        ],
    },
    {
        "id": "dev_mhop_01",
        "split": "dev",
        "main_group": "multi_hop",
        "question": "Synthesizing Table SPM.1 (page 37) and paragraph A.1 (page 19): how does the current 1.1°C warming level relate to the urgency of achieving the 43% GHG cut by 2030?",
        "answerable": True,
        "expected_answer": "With global warming already having reached 1.1°C, immediate and rapid emission reductions of 43% by 2030 are necessary to prevent exceeding the 1.5°C threshold in the near term.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 19,
                "table_or_fig": "SPM A.1",
                "bbox": [70, 100, 540, 250],
                "evidence": "Warming reached 1.1°C in 2011-2020.",
                "units": "°C",
            },
            {
                "source_id": "ipcc-ar6-syr",
                "page": 37,
                "table_or_fig": "Table SPM.1",
                "bbox": [79, 78, 538, 149],
                "evidence": "43% reduction needed by 2030.",
                "units": "%",
            },
        ],
    },
    {
        "id": "dev_mhop_02",
        "split": "dev",
        "main_group": "multi_hop",
        "question": "Combining Figure SPM.1(a) (page 23) and Section 3.2 text (page 92): what is the projected impact on coral reefs if global warming reaches 1.5°C versus 2°C?",
        "answerable": True,
        "expected_answer": "Coral reefs are projected to decline by a further 70-90% at 1.5°C of warming, with larger losses (>99%) at 2°C.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 23,
                "table_or_fig": "Figure SPM.1(a)",
                "bbox": [70, 75, 540, 350],
                "evidence": "Coral reefs show severe losses already observed.",
                "units": "Impact status",
            },
            {
                "source_id": "ipcc-ar6-syr",
                "page": 92,
                "table_or_fig": "Section 3.2 text",
                "bbox": [70, 100, 540, 300],
                "evidence": "Coral reefs decline 70-90% at 1.5°C and >99% at 2°C.",
                "units": "% decline",
            },
        ],
    },
    {
        "id": "dev_mhop_03",
        "split": "dev",
        "main_group": "multi_hop",
        "question": "Comparing Figure SPM.7(a) (page 43) and Section 4.5 text (page 118): why does nuclear energy show smaller mitigation potential compared to solar and wind by 2030?",
        "answerable": True,
        "expected_answer": "Nuclear power has longer lead times for construction and higher levelized costs compared to the rapidly scalable and cost-effective solar and wind options.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 43,
                "table_or_fig": "Figure SPM.7(a)",
                "bbox": [70, 75, 540, 450],
                "evidence": "Nuclear bar is significantly shorter than wind and solar.",
                "units": "Bar length comparison",
            },
            {
                "source_id": "ipcc-ar6-syr",
                "page": 118,
                "table_or_fig": "Section 4.5 text",
                "bbox": [70, 100, 540, 300],
                "evidence": "Nuclear energy is constrained by construction time and financing.",
                "units": "Feasibility constraints",
            },
        ],
    },
    {
        "id": "dev_text_01",
        "split": "dev",
        "main_group": "text",
        "question": "In paragraph A.1.2 on page 19, what were the estimated global surface temperature increases over land versus over the ocean in 2011-2020 relative to 1850-1900?",
        "answerable": True,
        "expected_answer": "1.59°C [1.34°C to 1.83°C] over land and 0.88°C [0.68°C to 1.01°C] over the ocean.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 19,
                "table_or_fig": "SPM A.1.2 text",
                "bbox": [70, 200, 540, 350],
                "evidence": "Global surface temperature was 1.59 [1.34 to 1.83] °C higher over land and 0.88 [0.68 to 1.01] °C higher over the ocean.",
                "units": "°C",
            }
        ],
    },
    {
        "id": "dev_text_02",
        "split": "dev",
        "main_group": "text",
        "question": "According to paragraph B.2.2 on page 31, what happens to the frequency and intensity of heavy precipitation events with every additional increment of global warming?",
        "answerable": True,
        "expected_answer": "Heavy precipitation events are projected to intensify and become more frequent over most regions with additional global warming.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 31,
                "table_or_fig": "SPM B.2.2 text",
                "bbox": [70, 100, 540, 250],
                "evidence": "Heavy precipitation events will intensify and become more frequent in most regions with additional warming.",
                "units": "Statement quote",
            }
        ],
    },
    {
        "id": "dev_text_03",
        "split": "dev",
        "main_group": "text",
        "question": "What is the primary message of paragraph C.2 on page 38 concerning deep, rapid, and sustained greenhouse gas emissions reduction?",
        "answerable": True,
        "expected_answer": "Deep, rapid, and sustained GHG emissions reduction would lead to a discernible slowdown in global warming within around two decades, and to discernible changes in atmospheric composition within a few years.",
        "gold": [
            {
                "source_id": "ipcc-ar6-syr",
                "page": 38,
                "table_or_fig": "SPM C.2 text",
                "bbox": [70, 100, 540, 300],
                "evidence": "Deep, rapid, and sustained emissions reduction would lead to a discernible slowdown in global warming within around two decades.",
                "units": "Statement quote",
            }
        ],
    },
    {
        "id": "dev_unans_01",
        "split": "dev",
        "main_group": "unanswerable",
        "question": "What is the exact price in Euro of a bus ticket in Paris specified in the IPCC AR6 Synthesis Report transport section?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Searched document for bus ticket prices in Paris. The report assesses macro transport mitigation potentials, not local public transport fares.",
    },
    {
        "id": "dev_unans_02",
        "split": "dev",
        "main_group": "unanswerable",
        "question": "According to Table SPM.1, what is the exact percentage increase in tomato crop yield in Spain in 2040?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Table SPM.1 only contains global aggregate GHG and CO2 percentage emission reductions, not crop-specific yields in Spain.",
    },
    {
        "id": "dev_unans_03",
        "split": "dev",
        "main_group": "unanswerable",
        "question": "What is the secret recipe of the culinary dinner served to delegates at the IPCC approval plenary in Interlaken?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Culinary recipes and catering menus do not exist in IPCC scientific assessment reports.",
    },
]