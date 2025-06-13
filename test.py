def crop_single_segment_between_stops(segments, board_stop, alight_stop, stops_dict):
    if not segments or not segments[0]:
        return []
    poly = segments[0]

    # Витягуємо реальні координати зупинок
    la_b, lo_b, _ = stops_dict.get(board_stop, (None, None, None))
    la_a, lo_a, _ = stops_dict.get(alight_stop, (None, None, None))
    if la_b is None or la_a is None:
        return []

    # Поріг у градусах (приблизно): 1° ≈ 111 000 м за широтою
    threshold_sq = (CLIP_THRESHOLD_METERS / 111000.0) ** 2

    # Шукаємо перший індекс bi, де точка траси попадає в радіус ≤20 м від board_stop
    bi = None
    for idx, (plat, plon) in enumerate(poly):
        d_sq = (plat - la_b)**2 + (plon - lo_b)**2
        if d_sq <= threshold_sq:
            bi = idx
            break
    # Якщо не знайшли жодної точки у радіусі 20 м, беремо closest_index як fallback
    if bi is None:
        bi = find_closest_index(poly, (la_b, lo_b))

    # Аналогічно для alight_stop: шукаємо ОСТАННІЙ індекс ai, де відстань ≤20 м
    ai = None
    for idx in range(len(poly)-1, -1, -1):
        plat, plon = poly[idx]
        d_sq = (plat - la_a)**2 + (plon - lo_a)**2
        if d_sq <= threshold_sq:
            ai = idx
            break
    # Якщо не знайшли — fallback до closest_index
    if ai is None:
        ai = find_closest_index(poly, (la_a, lo_a))

    if bi is None or ai is None:
        return []

    # Якщо порядок index b → index a нормальний
    if bi <= ai:
        cropped_pts = poly[bi: ai + 1]
    else:
        # Якщо траса повертається «назад», обрізаємо й розвертаємо
        cropped_pts = poly[ai: bi + 1]
        cropped_pts.reverse()

    # Примусово ставимо першу й останню точку зрізаного списку
    cropped_pts[0] = (la_b, lo_b)
    cropped_pts[-1] = (la_a, lo_a)

    return [cropped_pts]