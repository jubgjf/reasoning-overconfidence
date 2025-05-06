import json
import matplotlib.pyplot as plt
import seaborn as sns
import random
from itertools import product

from confidence.data import TimeTablingData


def generate_problem():
    """生成一个保证存在可行解的排课问题"""
    n_courses = 10
    n_teachers = 5
    n_times = 10
    n_rooms = 10

    # 生成基础可行解
    solution = {}
    used_tr = set()  # 已占用的 (时间, 教室)
    used_tt = set()  # 已占用的 (教师, 时间)
    course_options = {}

    for course in range(n_courses):
        while True:
            time = random.randint(0, n_times - 1)
            room = random.randint(0, n_rooms - 1)
            teacher = random.randint(0, n_teachers - 1)
            if (time, room) not in used_tr and (teacher, time) not in used_tt:
                # 记录可行解
                solution[f"Course{course}"] = {"time": time, "room": room, "teacher": teacher}
                used_tr.add((time, room))
                used_tt.add((teacher, time))
                # 生成课程选项（至少包含可行解的值，并随机扩展）
                possible_times = [time]
                if random.random() < 0.5 and time + 1 < n_times:
                    possible_times.append(time + 1)
                possible_rooms = [room]
                possible_teachers = [teacher]
                course_options[f"Course{course}"] = {
                    "times": possible_times,
                    "rooms": possible_rooms,
                    "teachers": possible_teachers,
                }
                break
    return course_options, solution


def find_all_solutions(course_options):
    """遍历所有可能组合，返回满足约束的解"""
    course_list = []
    for course in course_options:
        co = course_options[course]
        assignments = []
        for t in co["times"]:
            for r in co["rooms"]:
                for p in co["teachers"]:
                    assignments.append({"course": course, "time": t, "room": r, "teacher": p})
        course_list.append(assignments)

    all_combinations = product(*course_list)
    solutions = []

    for combo in all_combinations:
        time_room = set()
        teacher_time = set()
        valid = True
        for assign in combo:
            t = assign["time"]
            r = assign["room"]
            p = assign["teacher"]
            if (t, r) in time_room or (p, t) in teacher_time:
                valid = False
                break
            time_room.add((t, r))
            teacher_time.add((p, t))
        if valid:
            solutions.append(combo)
    return solutions


with open("./dataset/timetabling.jsonl", "w") as f:
    question_id, max_questions_per_bin = 0, 800
    solution_bin = {i: [] for i in range(1, 8 + 1)}  # 1: 50~100, 2: 100~150, ..., 7: 350~400, 8: >400
    while True:
        course_options, _ = generate_problem()
        all_solutions = find_all_solutions(course_options)

        question = "Constraints:\n"
        for course in course_options:
            opts = course_options[course]
            question += f"- {course} : Time {opts['times']}, Room {opts['rooms']}, Teacher {opts['teachers']}\n"
        question += "- Multiple courses cannot be scheduled in the same time slot and room.\n"
        question += "- A teacher can only teach one course at a time.\n"

        answers = f"Total {len(all_solutions)} solutions\n\n"
        for idx, sol in enumerate(all_solutions, 1):
            answers += f"Solution {idx}:\n"
            answers += "| Course  | Time  | Room  | Teacher  |\n"
            answers += "|---------|-------|-------|----------|\n"
            for assign in sol:
                answers += f"| {assign['course']} | T{assign['time']}    | R{assign['room']}    | P{assign['teacher']}       |\n"
            answers += "\n"

        data = TimeTablingData(
            question_id=question_id, question=question, answers={"0": answers}, answer_count=len(all_solutions)
        )

        if len(all_solutions) < 50:
            continue
        elif len(all_solutions) < 400:
            solution_bin[len(all_solutions) // 50].append(data)
        else:
            solution_bin[8].append(data)

        can_break = True
        for k, solutions in solution_bin.items():
            if len(solutions) < max_questions_per_bin:
                can_break = False
            elif len(solutions) > max_questions_per_bin * 5:
                solution_bin[k] = []
        if can_break:
            print("-------------")
            print({k: len(v) for k, v in solution_bin.items()})
            break

        print({k: len(v) for k, v in solution_bin.items()})

    question_id = 0
    for k, solutions in solution_bin.items():
        solutions = solutions[:max_questions_per_bin]
        for solution in solutions:
            solution.question_id = question_id
            f.write(json.dumps(solution.model_dump(), ensure_ascii=False) + "\n")
            question_id += 1

with open("./dataset/timetabling.jsonl") as f:
    dataset = [json.loads(line) for line in f.readlines()]
answer_counts = [data["answer_count"] for data in dataset]
answer_counts_bins = [int(x // 50) if int(x // 50) < 8 else 8 for x in answer_counts]
sns.histplot(answer_counts_bins, bins=8)
plt.xlabel("Number of Solutions")
plt.ylabel("Frequency")
plt.title("Distribution of Number of Solutions")
plt.show()
