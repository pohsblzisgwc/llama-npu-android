// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 The llama.cpp Authors

package com.example.llama.ui

import android.graphics.Color
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.llama.R
import com.example.llama.data.Conversation
import com.google.android.material.card.MaterialCardView
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class ConversationAdapter(
    private val conversations: MutableList<Conversation>,
    private var activeConversationId: String,
    private val onConversationClick: (Conversation) -> Unit,
    private val onConversationDelete: (Conversation) -> Unit
) : RecyclerView.Adapter<ConversationAdapter.ViewHolder>() {

    private val dateFormat = SimpleDateFormat("MM-dd HH:mm", Locale.getDefault())

    fun setActiveConversation(id: String) {
        activeConversationId = id
        notifyDataSetChanged()
    }

    fun updateData(newItems: List<Conversation>) {
        conversations.clear()
        conversations.addAll(newItems)
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_conversation, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val conv = conversations[position]
        holder.tvTitle.text = conv.title
        val lastMsg = conv.messages.lastOrNull()?.content?.trim() ?: "尚无对话消息"
        holder.tvPreview.text = lastMsg
        holder.tvTime.text = dateFormat.format(Date(conv.updatedAt))

        val isActive = conv.id == activeConversationId
        if (isActive) {
            holder.card.strokeColor = Color.parseColor("#3B82F6")
            holder.card.setCardBackgroundColor(Color.parseColor("#1E293B"))
        } else {
            holder.card.strokeColor = Color.parseColor("#2A2E39")
            holder.card.setCardBackgroundColor(Color.parseColor("#141824"))
        }

        holder.itemView.setOnClickListener {
            onConversationClick(conv)
        }

        holder.btnDelete.setOnClickListener {
            onConversationDelete(conv)
        }
    }

    override fun getItemCount(): Int = conversations.size

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val card: MaterialCardView = view.findViewById(R.id.card_conversation)
        val tvTitle: TextView = view.findViewById(R.id.tv_conv_title)
        val tvPreview: TextView = view.findViewById(R.id.tv_conv_preview)
        val tvTime: TextView = view.findViewById(R.id.tv_conv_time)
        val btnDelete: ImageButton = view.findViewById(R.id.btn_delete_conv)
    }
}
