// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 The llama.cpp Authors

package com.example.llama.data

import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    var content: String,
    val isUser: Boolean,
    val timestamp: Long = System.currentTimeMillis(),
    var imageUri: String? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id)
        put("content", content)
        put("isUser", isUser)
        put("timestamp", timestamp)
        if (imageUri != null) {
            put("imageUri", imageUri)
        }
    }

    companion object {
        fun fromJson(json: JSONObject): ChatMessage = ChatMessage(
            id = json.optString("id", UUID.randomUUID().toString()),
            content = json.optString("content", ""),
            isUser = json.optBoolean("isUser", false),
            timestamp = json.optLong("timestamp", System.currentTimeMillis()),
            imageUri = if (json.has("imageUri") && !json.isNull("imageUri")) json.getString("imageUri") else null
        )
    }
}

data class Conversation(
    val id: String = UUID.randomUUID().toString(),
    var title: String = "新对话",
    val createdAt: Long = System.currentTimeMillis(),
    var updatedAt: Long = System.currentTimeMillis(),
    val messages: MutableList<ChatMessage> = mutableListOf()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id)
        put("title", title)
        put("createdAt", createdAt)
        put("updatedAt", updatedAt)
        val arr = JSONArray()
        messages.forEach { arr.put(it.toJson()) }
        put("messages", arr)
    }

    companion object {
        fun fromJson(json: JSONObject): Conversation {
            val conv = Conversation(
                id = json.optString("id", UUID.randomUUID().toString()),
                title = json.optString("title", "新对话"),
                createdAt = json.optLong("createdAt", System.currentTimeMillis()),
                updatedAt = json.optLong("updatedAt", System.currentTimeMillis())
            )
            val arr = json.optJSONArray("messages")
            if (arr != null) {
                for (i in 0 until arr.length()) {
                    val msgObj = arr.optJSONObject(i)
                    if (msgObj != null) {
                        conv.messages.add(ChatMessage.fromJson(msgObj))
                    }
                }
            }
            return conv
        }
    }
}
