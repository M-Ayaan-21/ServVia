from django.contrib import admin

from .models import UserRemedyOutcome


@admin.register(UserRemedyOutcome)
class UserRemedyOutcomeAdmin(admin.ModelAdmin):
    list_display = [
        "user_email", "remedy_name", "condition_treated",
        "efficacy_score", "improvement_noted", "tried_at",
    ]
    list_filter = ["efficacy_score", "improvement_noted"]
    search_fields = ["user_email", "remedy_name", "condition_treated"]
    ordering = ["-tried_at"]
    readonly_fields = ["tried_at"]
