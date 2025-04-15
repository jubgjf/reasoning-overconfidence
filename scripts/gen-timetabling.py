import json
import random

from confidence.data import TimeTablingData


def gen_timetabling() -> tuple[str, str]:
    generated_question = ""
    answer_example = ""

    # 定义基本参数
    num_courses = 5  # 课程数量
    num_teachers = 3  # 教师数量
    num_students = 30  # 学生数量
    num_rooms = 2  # 教室数量
    num_time_slots = 3  # 可用时间段数量
    room_capacities = [20, 15]  # 每个教室的容量

    # 随机生成课程，教师，学生，和时间段的分配
    courses = [f"Course{i}" for i in range(1, num_courses + 1)]
    teachers = [f"Teacher{i}" for i in range(1, num_teachers + 1)]
    students = [f"Student{i}" for i in range(1, num_students + 1)]
    rooms = [f"Room{i}" for i in range(1, num_rooms + 1)]
    time_slots = [f"TimeSlot{i}" for i in range(1, num_time_slots + 1)]

    # 随机分配每门课程给教师
    course_teacher_mapping = {}
    for course in courses:
        course_teacher_mapping[course] = random.choice(teachers)

    # 随机分配每门课程给学生（每个学生选修随机课程）
    student_course_mapping = {}
    for student in students:
        student_course_mapping[student] = random.sample(courses, random.randint(1, 3))  # 每个学生选修1到3门课程

    # 随机分配每门课程一个教室和时间段
    course_schedule = {}
    for course in courses:
        course_schedule[course] = {
            "teacher": course_teacher_mapping[course],
            "room": random.choice(rooms),
            "time_slot": random.choice(time_slots),
        }

    # 打印出生成的排课信息
    generated_question += "Courses schedule:\n"
    for course, schedule in course_schedule.items():
        generated_question += (
            f"{course}: Teacher={schedule['teacher']}, Room={schedule['room']}, TimeSlot={schedule['time_slot']}\n"
        )
    generated_question += "\n"

    # 生成约束条件
    constraints = []

    # 确保每门课程都有安排
    for course in courses:
        constraints.append(f"{course} must be scheduled at some time slot and room.")

    # 确保每个教师在每个时间段只安排一门课程
    for teacher in teachers:
        constraints.append(f"{teacher} can only teach one course at a time.")

    # 确保每个教室的学生数量不超过教室容量
    for room, capacity in zip(rooms, room_capacities):
        enrolled_students_in_room = [
            student
            for student, courses in student_course_mapping.items()
            if any(course_schedule[course]["room"] == room for course in courses)
        ]
        constraints.append(
            f"Room {room} can accommodate up to {capacity} students. Current enrollment: {len(enrolled_students_in_room)} students."
        )

    # 确保每个学生在不同课程的时间段不会冲突
    for student, enrolled_courses in student_course_mapping.items():
        for i, course1 in enumerate(enrolled_courses):
            for course2 in enrolled_courses[i + 1 :]:
                if course_schedule[course1]["time_slot"] == course_schedule[course2]["time_slot"]:
                    constraints.append(f"{student} cannot have {course1} and {course2} at the same time.")

    # 输出约束条件
    generated_question += "Constraints:\n"
    for constraint in constraints:
        generated_question += constraint + "\n"

    # 可行解输出（部分）
    for course, schedule in course_schedule.items():
        answer_example += (
            f"{course} is scheduled in {schedule['room']} at {schedule['time_slot']} with {schedule['teacher']}."
        )

    return generated_question, answer_example


if __name__ == "__main__":
    with open("./dataset/timetabling.jsonl", "w") as f:
        for i in range(200):
            q, a = gen_timetabling()
            data = TimeTablingData(question_id=i, question=q, answer_example=a)
            f.write(json.dumps(data.model_dump(), ensure_ascii=False) + "\n")
