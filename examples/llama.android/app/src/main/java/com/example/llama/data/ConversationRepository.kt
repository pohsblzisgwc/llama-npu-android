// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 The llama.cpp Authors

package com.example.llama.data

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.util.UUID

class ConversationRepository(private val context: Context) {

    private val convDir: File by lazy {
        File(context.filesDir, "conversations").also {
            if (!it.exists()) it.mkdirs()
        }
    }

    @Synchronized
    fun getAllConversations(): List<Conversation> {
        val list = mutableListOf<Conversation>()
        val files = convDir.listFiles { _, name -> name.endsWith(".json") } ?: emptyArray()
        for (file in files) {
            try {
                val jsonStr = file.readText()
                val conv = Conversation.fromJson(JSONObject(jsonStr))
                list.add(conv)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to load conversation file: ${file.name}", e)
            }
        }
        return list.sortedByDescending { it.updatedAt }
    }

    private fun isValidId(id: String): Boolean {
        return id.isNotBlank() && id.matches(Regex("^[a-zA-Z0-9_-]+$"))
    }

    @Synchronized
    fun getConversation(id: String): Conversation? {
        if (!isValidId(id)) return null
        val file = File(convDir, "$id.json")
        if (!file.exists()) return null
        return try {
            val jsonStr = file.readText()
            Conversation.fromJson(JSONObject(jsonStr))
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load conversation $id", e)
            null
        }
    }

    @Synchronized
    fun saveConversation(conv: Conversation) {
        if (!isValidId(conv.id)) return
        try {
            val file = File(convDir, "${conv.id}.json")
            val jsonStr = conv.toJson().toString()
            file.writeText(jsonStr)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save conversation ${conv.id}", e)
        }
    }

    @Synchronized
    fun deleteConversation(id: String) {
        if (!isValidId(id)) return
        try {
            val file = File(convDir, "$id.json")
            if (file.exists()) file.delete()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to delete conversation $id", e)
        }
    }

    fun createNewConversation(title: String = "新对话"): Conversation {
        val conv = Conversation(
            id = UUID.randomUUID().toString(),
            title = title,
            createdAt = System.currentTimeMillis(),
            updatedAt = System.currentTimeMillis()
        )
        saveConversation(conv)
        return conv
    }

    companion object {
        private const val TAG = "ConversationRepo"
    }
}
