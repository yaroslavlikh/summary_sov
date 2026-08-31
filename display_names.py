DISPLAY_NAMES = {
    "Beeeeeeeee1E": "Ярик Комаров",
    "geniymover": "Тимофей",
    "Goshanka200": "Игорь",
    "Ivanushka_Internetional": "Ваня",
    "p34108": "Влад",
    "presccode80": "Ivjenin",
    "Sahaaai": "Сахаи / Саша Резаков",
    "Super_rumit": "Гордей",
    "tigmen": "tigmen / Саша Тигмен",
    "yaroslavlikh": "Ярик Лихачев",
}


def resolve_display_name(username, fallback):
    if not username:
        return fallback
    return DISPLAY_NAMES.get(username.lstrip('@'), fallback)
