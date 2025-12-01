from django.db import models
from django.utils import timezone

class UserProfile(models.Model):
    email = models.EmailField(unique=True, db_index=True)
    first_name = models.CharField(max_length=100, blank=True, null=True)
    last_name = models.CharField(max_length=100, blank=True, null=True)
    
    allergies = models.TextField(blank=True, null=True)
    medical_conditions = models.TextField(blank=True, null=True)
    current_medications = models.TextField(blank=True, null=True)
    
    is_profile_complete = models.BooleanField(default=False)
    profile_completed_at = models.DateTimeField(blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_login = models.DateTimeField(default=timezone.now)
    
    class Meta:
        db_table = 'user_profiles'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.email} - {self.first_name or 'Unknown'}"
    
    def get_allergies_list(self):
        if self.allergies:
            return [a.strip() for a in self.allergies.split(',') if a.strip()]
        return []
    
    def get_conditions_list(self):
        if self.medical_conditions:
            return [c.strip() for c in self.medical_conditions.split(',') if c.strip()]
        return []
    
    def get_medications_list(self):
        if self.current_medications:
            return [m.strip() for m in self.current_medications.split(',') if m.strip()]
        return []
    
    def mark_profile_complete(self):
        self.is_profile_complete = True
        self.profile_completed_at = timezone.now()
        self.save()
