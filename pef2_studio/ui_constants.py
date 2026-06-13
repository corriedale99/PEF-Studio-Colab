from __future__ import annotations


SYMBOL_CATEGORY_UI = {
    "drop": {
        "label": "読まない記号",
        "description": "TTSでは読み上げない予定の記号です。",
        "css_class": "sym-drop",
        "color_var": "--sym-drop-color",
        "background_var": "--sym-drop-bg",
    },
    "pause_s": {
        "label": "短い息継ぎ",
        "description": "短い間・軽い息継ぎとして扱う予定の記号です。",
        "css_class": "sym-pause_s",
        "color_var": "--sym-pause-s-color",
        "background_var": "--sym-pause-s-bg",
    },
    "pause_m": {
        "label": "少し長い間",
        "description": "少し長めの間として扱う予定の記号です。",
        "css_class": "sym-pause_m",
        "color_var": "--sym-pause-m-color",
        "background_var": "--sym-pause-m-bg",
    },
    "keep": {
        "label": "通常記号",
        "description": "通常の記号として表示します。",
        "css_class": "sym-keep",
        "color_var": "--sym-keep-color",
        "background_var": "--sym-keep-bg",
    },
    "reading_annotation": {
        "label": "読み指定",
        "description": "原文《読み》 の形で表示している読み指定です。新しく読みを指定するときは、｜原文《読み》 の形で範囲を明示できます。キーボードで入力しやすい半角 | も使えます。",
        "css_class": "reading-annotation",
        "color_var": "--reading-color",
        "background_var": "--reading-bg",
    },
}
